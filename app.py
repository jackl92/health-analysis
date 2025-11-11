import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Voice Health Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib to non-interactive backend for faster rendering
plt.switch_backend('Agg')
plt.ioff()  # Turn off interactive mode

# Color palettes
HEALTH_COLORS = {'healthy': '#66c2a5', 'unhealthy': '#fc8d62'}
GENDER_COLORS = {'male': '#8da0cb', 'female': '#e78ac3'}

# Load data with caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    df = pd.read_csv('data/cleaned_health_data.csv')
    return df

# Cache plotting functions
@st.cache_data
def create_age_distribution_plot(data, health_status_list):
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in health_status_list:
        subset = data[data['health_status'] == status]['age']
        ax.hist(subset, bins=30, alpha=0.6, label=status, color=HEALTH_COLORS[status])
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Age Distribution by Health Status', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

@st.cache_data
def create_health_gender_plot(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    gender_health = data.groupby(['health_status', 'gender']).size().unstack()
    gender_health.plot(kind='bar', ax=ax, color=[GENDER_COLORS['female'], GENDER_COLORS['male']])
    ax.set_xlabel('Health Status', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Health Status Distribution by Gender', fontsize=14)
    ax.legend(title='Gender')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=0)
    return fig

@st.cache_data
def create_feature_histogram(data, feature, health_status_list):
    fig, ax = plt.subplots(figsize=(10, 6))
    for status in health_status_list:
        subset = data[data['health_status'] == status][feature]
        ax.hist(subset, bins=30, alpha=0.6, label=status, color=HEALTH_COLORS[status], edgecolor='black')
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    return fig

@st.cache_data
def create_mfcc_correlation_matrix(data, mfcc_cols):
    fig, ax = plt.subplots(figsize=(14, 10))
    correlation_matrix = data[mfcc_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('MFCC Correlation Matrix', fontsize=16, pad=20)
    return fig

# Main app
def main():
    st.title("🎤 Voice Health Analysis Dashboard")
    st.markdown("### Interactive Exploration of Voice Features & Health Status")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("🎛️ Filter Options")
    
    # Filters
    health_status = st.sidebar.multiselect(
        "Health Status",
        options=df['health_status'].unique(),
        default=df['health_status'].unique()
    )
    
    gender = st.sidebar.multiselect(
        "Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )
    
    age_range = st.sidebar.slider(
        "Age Range",
        int(df['age'].min()),
        int(df['age'].max()),
        (int(df['age'].min()), int(df['age'].max()))
    )
    
    # Filter data
    filtered_df = df[
        (df['health_status'].isin(health_status)) &
        (df['gender'].isin(gender)) &
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1])
    ]
    
    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filtered Dataset Stats")
    st.sidebar.metric("Total Samples", len(filtered_df))
    st.sidebar.metric("Healthy", len(filtered_df[filtered_df['health_status'] == 'healthy']))
    st.sidebar.metric("Unhealthy", len(filtered_df[filtered_df['health_status'] == 'unhealthy']))
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", 
        "👥 Demographics", 
        "🎵 Voice Features", 
        "🔢 MFCCs",
        "📊 Custom Analysis"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 3)
        with col3:
            healthy_pct = (df['health_status'] == 'healthy').sum() / len(df) * 100
            st.metric("Healthy %", f"{healthy_pct:.1f}%")
        with col4:
            female_pct = (df['gender'] == 'female').sum() / len(df) * 100
            st.metric("Female %", f"{female_pct:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Key Statistics")
            st.dataframe(filtered_df.describe().T.style.format("{:.2f}"), height=400)
        
        with col2:
            st.subheader("⚠️ Critical Findings")
            st.markdown("""
            **Age Confounding:**
            - Healthy median age: **22 years**
            - Unhealthy median age: **52 years**
            - **30-year age gap** is a major confounding variable
            
            **Class Distribution:**
            - 66% unhealthy vs 34% healthy (2:1 ratio)
            - Class imbalance handled in ML models
            
            **Gender Patterns:**
            - Males show dramatic voice changes when unhealthy
            - Females remain relatively stable
            """)
            
            st.info("💡 **Insight**: The large age gap means we cannot definitively separate health effects from natural aging.")
    
    # Tab 2: Demographics
    with tab2:
        st.header("👥 Demographics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution by Health Status")
            fig = create_age_distribution_plot(filtered_df, filtered_df['health_status'].unique())
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            # Age stats
            st.markdown("**Age Statistics:**")
            for status in filtered_df['health_status'].unique():
                median_age = filtered_df[filtered_df['health_status'] == status]['age'].median()
                st.write(f"- {status.capitalize()}: Median = {median_age:.0f} years")
        
        with col2:
            st.subheader("Health Status by Gender")
            fig = create_health_gender_plot(filtered_df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            # Gender stats
            st.markdown("**Gender Distribution:**")
            for gender_val in filtered_df['gender'].unique():
                count = len(filtered_df[filtered_df['gender'] == gender_val])
                pct = count / len(filtered_df) * 100
                st.write(f"- {gender_val.capitalize()}: {count} ({pct:.1f}%)")
    
    # Tab 3: Voice Features
    with tab3:
        st.header("🎵 Voice Characteristics")
        
        # Feature selector
        voice_features = [
            'spectral_centroid', 'spectral_bandwidth', 'rolloff', 
            'rmse', 'zero_crossing_rate', 'chroma_stft'
        ]
        
        selected_feature = st.selectbox(
            "Select Voice Feature to Explore:",
            voice_features,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Feature descriptions
        feature_info = {
            'spectral_centroid': "**Voice brightness** - Center of mass of frequencies. Higher = brighter, clearer voice.",
            'spectral_bandwidth': "**Frequency spread** - How scattered frequencies are. Higher = more irregular patterns.",
            'rolloff': "**Energy distribution** - Frequency below which 85% of energy exists. Indicates breathiness.",
            'rmse': "**Voice energy/loudness** - Root mean square energy. Higher = louder, more energetic.",
            'zero_crossing_rate': "**Voice noisiness** - How often signal crosses zero. Higher = more breathiness/noise.",
            'chroma_stft': "**Pitch class distribution** - Energy distribution across musical notes."
        }
        
        st.info(feature_info[selected_feature])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_feature.replace('_', ' ').title()} by Health Status")
            fig = create_feature_histogram(filtered_df, selected_feature, filtered_df['health_status'].unique())
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
            # Statistics
            st.markdown("**Statistics by Health Status:**")
            for status in filtered_df['health_status'].unique():
                mean_val = filtered_df[filtered_df['health_status'] == status][selected_feature].mean()
                std_val = filtered_df[filtered_df['health_status'] == status][selected_feature].std()
                st.write(f"- {status.capitalize()}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
        
        with col2:
            st.subheader(f"{selected_feature.replace('_', ' ').title()} by Gender & Health")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Healthy subplot
            healthy_data = filtered_df[filtered_df['health_status'] == 'healthy']
            for gender_val in healthy_data['gender'].unique():
                data = healthy_data[healthy_data['gender'] == gender_val][selected_feature]
                ax1.hist(data, bins=20, alpha=0.6, label=gender_val, color=GENDER_COLORS[gender_val], edgecolor='black')
            ax1.set_title('Healthy Individuals', fontsize=12)
            ax1.set_xlabel(selected_feature.replace('_', ' ').title(), fontsize=11)
            ax1.set_ylabel('Count', fontsize=11)
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Unhealthy subplot
            unhealthy_data = filtered_df[filtered_df['health_status'] == 'unhealthy']
            for gender_val in unhealthy_data['gender'].unique():
                data = unhealthy_data[unhealthy_data['gender'] == gender_val][selected_feature]
                ax2.hist(data, bins=20, alpha=0.6, label=gender_val, color=GENDER_COLORS[gender_val], edgecolor='black')
            ax2.set_title('Unhealthy Individuals', fontsize=12)
            ax2.set_xlabel(selected_feature.replace('_', ' ').title(), fontsize=11)
            ax2.set_ylabel('Count', fontsize=11)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        st.markdown("---")
        st.subheader("📊 Box Plot Comparison")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        filtered_df.boxplot(column=selected_feature, by=['health_status', 'gender'], ax=ax)
        ax.set_xlabel('Health Status & Gender', fontsize=12)
        ax.set_ylabel(selected_feature.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{selected_feature.replace("_", " ").title()} by Health & Gender', fontsize=14)
        plt.suptitle('')  # Remove default title
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    # Tab 4: MFCCs
    with tab4:
        st.header("🔢 MFCC Feature Analysis")
        st.markdown("**Mel-Frequency Cepstral Coefficients** capture voice timbre in a way that mimics human hearing.")
        
        mfcc_cols = [col for col in df.columns if 'mfcc' in col]
        
        st.subheader("MFCC Correlation Matrix")
        st.markdown("Low correlations (< 0.4) indicate good feature diversity - each MFCC captures unique information.")
        
        with st.spinner("Generating correlation matrix..."):
            fig = create_mfcc_correlation_matrix(filtered_df, mfcc_cols)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("MFCC Feature Selector")
            selected_mfcc = st.selectbox("Select MFCC:", mfcc_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for status in filtered_df['health_status'].unique():
                data = filtered_df[filtered_df['health_status'] == status][selected_mfcc]
                ax.hist(data, bins=30, alpha=0.6, label=status, color=HEALTH_COLORS[status])
            ax.set_xlabel(selected_mfcc.upper(), fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{selected_mfcc.upper()} Distribution by Health Status', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        with col2:
            st.subheader("MFCC Summary Statistics")
            mfcc_stats = filtered_df.groupby('health_status')[selected_mfcc].describe()
            st.dataframe(mfcc_stats.style.format("{:.2f}"))
            
            st.markdown("**Interpretation:**")
            st.write(f"- MFCCs represent different aspects of voice timbre")
            st.write(f"- Each coefficient captures a unique spectral shape")
            st.write(f"- Values range from negative to positive")
            st.write(f"- Low inter-MFCC correlations = good feature engineering")
    
    # Tab 5: Custom Analysis
    with tab5:
        st.header("📊 Custom Analysis")
        st.markdown("Create your own visualizations by selecting features to compare.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox(
                "X-Axis Feature:",
                df.select_dtypes(include=[np.number]).columns.tolist(),
                index=df.select_dtypes(include=[np.number]).columns.tolist().index('age')
            )
        
        with col2:
            y_feature = st.selectbox(
                "Y-Axis Feature:",
                df.select_dtypes(include=[np.number]).columns.tolist(),
                index=df.select_dtypes(include=[np.number]).columns.tolist().index('spectral_centroid')
            )
        
        plot_type = st.radio("Plot Type:", ["Scatter Plot", "Hexbin Plot", "Box Plot"])
        
        st.markdown("---")
        
        if plot_type == "Scatter Plot":
            fig, ax = plt.subplots(figsize=(12, 8))
            for status in filtered_df['health_status'].unique():
                data = filtered_df[filtered_df['health_status'] == status]
                ax.scatter(data[x_feature], data[y_feature], 
                          label=status, alpha=0.6, s=50, color=HEALTH_COLORS[status])
            ax.set_xlabel(x_feature.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_feature.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        elif plot_type == "Hexbin Plot":
            fig, ax = plt.subplots(figsize=(12, 8))
            hexbin = ax.hexbin(filtered_df[x_feature], filtered_df[y_feature], 
                               gridsize=20, cmap='YlOrRd', mincnt=1)
            ax.set_xlabel(x_feature.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel(y_feature.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{y_feature.replace("_", " ").title()} vs {x_feature.replace("_", " ").title()} (Density)', fontsize=14)
            plt.colorbar(hexbin, ax=ax, label='Count')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            
        else:  # Box Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            filtered_df.boxplot(column=y_feature, by='health_status', ax=ax)
            ax.set_xlabel('Health Status', fontsize=12)
            ax.set_ylabel(y_feature.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{y_feature.replace("_", " ").title()} by Health Status', fontsize=14)
            plt.suptitle('')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        # Correlation analysis
        st.markdown("---")
        st.subheader("🔗 Correlation Analysis")
        correlation = filtered_df[[x_feature, y_feature]].corr().iloc[0, 1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation Coefficient", f"{correlation:.3f}")
        with col2:
            if abs(correlation) < 0.3:
                strength = "Weak"
            elif abs(correlation) < 0.7:
                strength = "Moderate"
            else:
                strength = "Strong"
            st.metric("Correlation Strength", strength)
        with col3:
            direction = "Positive" if correlation > 0 else "Negative"
            st.metric("Direction", direction)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎤 Voice Health Analysis Dashboard | Built with Streamlit</p>
        <p>Data: 2,037 voice recordings with 27 audio features</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
