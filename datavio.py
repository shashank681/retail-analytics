import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import numpy as np
import os
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Daily Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .upload-section {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .date-filter-section {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2196f3;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Data storage directory
DATA_STORAGE_DIR = Path('./data_storage')
DATA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DATA_FILE = DATA_STORAGE_DIR / "daily_data.json"

def save_daily_data(df, date_str, filename):
    """Save daily data to JSON storage with date organization"""
    try:
        # Load existing data
        if DAILY_DATA_FILE.exists():
            with open(DAILY_DATA_FILE, 'r', encoding='utf-8') as f:
                daily_data = json.load(f)
        else:
            daily_data = {}
        
        # Convert DataFrame to JSON-serializable format
        df_dict = df.to_dict('records')
        
        # Store data by date
        daily_data[date_str] = {
            'data': df_dict,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns)
        }
        
        # Save updated data
        with open(DAILY_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(daily_data, f, default=str, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"Error saving daily data: {str(e)}")
        return False

def load_daily_data():
    """Load all stored daily data"""
    try:
        if DAILY_DATA_FILE.exists():
            with open(DAILY_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading daily data: {str(e)}")
        return {}

def process_uploaded_file(uploaded_file, upload_date):
    """Process uploaded file and save with date organization"""
    try:
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Process data based on structure
        df = process_data(df)
        
        # Save with date organization
        success = save_daily_data(df, upload_date, uploaded_file.name)
        
        if success:
            st.success(f"✅ File uploaded successfully for date {upload_date}!")
            st.success(f"📊 Data saved: {len(df)} rows, {len(df.columns)} columns")
            return df
        else:
            st.error("❌ Failed to save data")
            return None
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def process_data(df):
    """Process and clean the uploaded data"""
    # Handle brand column if exists
    brand_columns = ['brand', 'Brand', 'BRAND']
    for col in brand_columns:
        if col in df.columns:
            # Clean brand data
            df[col] = df[col].astype(str).str.replace(r'^Brand:\s*', '', regex=True, case=False).str.strip()
            # Replace 'not found', 'nan', empty strings with 'Unknown'
            mask = df[col].str.lower().isin(['not found', 'nan', '']) | df[col].isna()
            df.loc[mask, col] = 'Unknown'
            break
    
    # Handle numeric columns
    numeric_columns = ['price', 'Price', 'rating', 'Rating', 'reviews', 'Reviews']
    for col in numeric_columns:
        if col in df.columns:
            # Remove 'not found' entries
            df = df[~df[col].astype(str).str.contains('not found|nan', case=False, na=False)]
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid ratings if rating column exists
    rating_cols = ['rating', 'Rating']
    for col in rating_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
            df = df[df[col] > 0]
            break
    
    return df

def get_data_summary(df):
    """Generate comprehensive data summary"""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'column_names': list(df.columns)
    }
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().round(2)
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary['categorical_summary'] = {}
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
    
    # Price statistics if price column exists
    price_columns = ['price', 'Price', 'cost_price', 'Cost price']
    for col in price_columns:
        if col in df.columns:
            summary['price_stats'] = {
                'min_price': df[col].min(),
                'max_price': df[col].max(),
                'avg_price': df[col].mean(),
                'median_price': df[col].median()
            }
            break
    
    # Rating statistics if rating column exists
    rating_columns = ['rating', 'Rating']
    for col in rating_columns:
        if col in df.columns:
            summary['rating_stats'] = {
                'min_rating': df[col].min(),
                'max_rating': df[col].max(),
                'avg_rating': df[col].mean(),
                'median_rating': df[col].median()
            }
            break
    
    return summary

def create_download_link(df, filename, file_format="csv"):
    """Create download link for data"""
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Download CSV</a>'
    else:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📥 Download Excel</a>'
    
    return href

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Daily Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    st.sidebar.header("📤 File Upload")
    
    # Upload section
    with st.sidebar.expander("📁 Upload New Data", expanded=True):
        st.markdown('<div class="upload-section">Upload your daily data file</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your daily data file (CSV or Excel format)"
        )
        
        # Date selection for upload
        upload_date = st.date_input(
            "Select Date for Data",
            value=datetime.now().date(),
            help="Choose the date for this data upload"
        ).strftime('%Y-%m-%d')
        
        # Process uploaded file
        if uploaded_file is not None:
            if st.button("📊 Process & Save Data"):
                with st.spinner("Processing uploaded file..."):
                    df = process_uploaded_file(uploaded_file, upload_date)
                    if df is not None:
                        st.rerun()  # Refresh to show new data
    
    # Load stored data
    daily_data = load_daily_data()
    
    # Date filter section
    st.sidebar.header("📅 Date Filter")
    
    if daily_data:
        with st.sidebar.expander("🗓️ Select Date", expanded=True):
            st.markdown('<div class="date-filter-section">Choose date to analyze</div>', unsafe_allow_html=True)
            
            available_dates = sorted(daily_data.keys(), reverse=True)
            selected_date = st.selectbox(
                "Available Dates",
                available_dates,
                help="Select a date to view its data"
            )
            
            if selected_date:
                # Load selected date data
                selected_data = daily_data[selected_date]
                df = pd.DataFrame(selected_data['data'])
                
                # Show basic info
                st.sidebar.success(f"✅ Data loaded for {selected_date}")
                st.sidebar.info(f"📊 {selected_data['row_count']} rows, {selected_data['column_count']} columns")
                st.sidebar.info(f"📁 File: {selected_data['filename']}")
                
                # Additional filters
                st.sidebar.subheader("🔍 Additional Filters")
                
                # Brand filter if brand column exists
                brand_columns = ['brand', 'Brand', 'BRAND']
                brand_col = None
                for col in brand_columns:
                    if col in df.columns:
                        brand_col = col
                        break
                
                if brand_col:
                    brands = ['All'] + sorted(df[brand_col].dropna().unique().tolist())
                    selected_brands = st.sidebar.multiselect(
                        "Select Brands",
                        brands,
                        default=['All']
                    )
                    if 'All' not in selected_brands and selected_brands:
                        df = df[df[brand_col].isin(selected_brands)].copy()
                
                # Rating filter if rating column exists
                rating_columns = ['rating', 'Rating']
                rating_col = None
                for col in rating_columns:
                    if col in df.columns:
                        rating_col = col
                        break
                
                if rating_col:
                    min_rating = float(df[rating_col].min())
                    max_rating = float(df[rating_col].max())
                    rating_range = st.sidebar.slider(
                        "Rating Range",
                        min_rating,
                        max_rating,
                        (min_rating, max_rating),
                        step=0.1
                    )
                    df = df[(df[rating_col] >= rating_range[0]) & (df[rating_col] <= rating_range[1])].copy()
    else:
        st.sidebar.info("📝 No data uploaded yet. Please upload a file to start analysis.")
        df = None
        selected_date = None
    
    # Main content area
    if df is not None and not df.empty:
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Analysis", "📋 Data Table", "📥 Download"])
        
        with tab1:
            st.subheader(f"📊 Data Summary for {selected_date}")
            
            # Get comprehensive summary
            summary = get_data_summary(df)
            
            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📋 Total Rows", summary['total_rows'])
            
            with col2:
                st.metric("📊 Total Columns", summary['total_columns'])
            
            with col3:
                if 'price_stats' in summary:
                    st.metric("💰 Avg Price", f"₹{summary['price_stats']['avg_price']:.2f}")
                else:
                    st.metric("🔢 Numeric Cols", len(df.select_dtypes(include=[np.number]).columns))
            
            with col4:
                if 'rating_stats' in summary:
                    st.metric("⭐ Avg Rating", f"{summary['rating_stats']['avg_rating']:.2f}")
                else:
                    st.metric("📝 Text Cols", len(df.select_dtypes(include=['object']).columns))
            
            # Column information
            st.subheader("📋 Column Information")
            col_info = []
            for col in df.columns:
                col_info.append({
                    'Column': col,
                    'Data Type': str(df[col].dtype),
                    'Non-Null Count': df[col].count(),
                    'Unique Values': df[col].nunique(),
                    'Sample Value': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
                })
            
            col_df = pd.DataFrame(col_info)
            st.dataframe(col_df, use_container_width=True)
            
            # Price statistics
            if 'price_stats' in summary:
                st.subheader("💰 Price Statistics")
                price_col1, price_col2, price_col3, price_col4 = st.columns(4)
                
                with price_col1:
                    st.metric("Min Price", f"₹{summary['price_stats']['min_price']:.2f}")
                with price_col2:
                    st.metric("Max Price", f"₹{summary['price_stats']['max_price']:.2f}")
                with price_col3:
                    st.metric("Average Price", f"₹{summary['price_stats']['avg_price']:.2f}")
                with price_col4:
                    st.metric("Median Price", f"₹{summary['price_stats']['median_price']:.2f}")
            
            # Rating statistics
            if 'rating_stats' in summary:
                st.subheader("⭐ Rating Statistics")
                rating_col1, rating_col2, rating_col3, rating_col4 = st.columns(4)
                
                with rating_col1:
                    st.metric("Min Rating", f"{summary['rating_stats']['min_rating']:.2f}")
                with rating_col2:
                    st.metric("Max Rating", f"{summary['rating_stats']['max_rating']:.2f}")
                with rating_col3:
                    st.metric("Average Rating", f"{summary['rating_stats']['avg_rating']:.2f}")
                with rating_col4:
                    st.metric("Median Rating", f"{summary['rating_stats']['median_rating']:.2f}")
        
        with tab2:
            st.subheader(f"📈 Data Analysis for {selected_date}")
            
            # Brand analysis if brand column exists
            brand_col = None
            for col in ['brand', 'Brand', 'BRAND']:
                if col in df.columns:
                    brand_col = col
                    break
            
            if brand_col:
                st.subheader("🏷️ Brand Analysis")
                brand_counts = df[brand_col].value_counts().head(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Brand distribution pie chart
                    fig_pie = px.pie(
                        values=brand_counts.values,
                        names=brand_counts.index,
                        title="Top 10 Brands Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Brand counts bar chart
                    fig_bar = px.bar(
                        x=brand_counts.index,
                        y=brand_counts.values,
                        title="Brand Counts",
                        labels={'x': 'Brand', 'y': 'Count'}
                    )
                    fig_bar.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Price analysis if price column exists
            price_col = None
            for col in ['price', 'Price', 'cost_price', 'Cost price']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col:
                st.subheader("💰 Price Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price distribution histogram
                    fig_hist = px.histogram(
                        df,
                        x=price_col,
                        title="Price Distribution",
                        nbins=30
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Price box plot
                    fig_box = px.box(
                        df,
                        y=price_col,
                        title="Price Box Plot"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Rating analysis if rating column exists
            rating_col = None
            for col in ['rating', 'Rating']:
                if col in df.columns:
                    rating_col = col
                    break
            
            if rating_col:
                st.subheader("⭐ Rating Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Rating distribution
                    fig_rating_hist = px.histogram(
                        df,
                        x=rating_col,
                        title="Rating Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_rating_hist, use_container_width=True)
                
                with col2:
                    # Average rating by brand (if both exist)
                    if brand_col:
                        avg_rating_by_brand = df.groupby(brand_col)[rating_col].mean().sort_values(ascending=False).head(10)
                        fig_brand_rating = px.bar(
                            x=avg_rating_by_brand.index,
                            y=avg_rating_by_brand.values,
                            title="Average Rating by Brand (Top 10)",
                            labels={'x': 'Brand', 'y': 'Average Rating'}
                        )
                        fig_brand_rating.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_brand_rating, use_container_width=True)
        
        with tab3:
            st.subheader(f"📋 Data Table for {selected_date}")
            
            # Search functionality
            search_term = st.text_input("🔍 Search in data", "")
            if search_term:
                mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                filtered_df = df[mask]
            else:
                filtered_df = df
            
            # Select columns to display
            display_columns = st.multiselect(
                "Select columns to display",
                df.columns.tolist(),
                default=df.columns.tolist()[:10] if len(df.columns) > 10 else df.columns.tolist()
            )
            
            if display_columns:
                display_df = filtered_df[display_columns]
                st.dataframe(display_df, use_container_width=True, height=400)
                
                st.info(f"Showing {len(filtered_df)} of {len(df)} records")
        
        with tab4:
            st.subheader("📥 Download Data")
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                csv_link = create_download_link(df, f"data_{selected_date}", "csv")
                st.markdown(csv_link, unsafe_allow_html=True)
            
            with col2:
                excel_link = create_download_link(df, f"data_{selected_date}", "excel")
                st.markdown(excel_link, unsafe_allow_html=True)
            
            # Summary report download
            st.subheader("📊 Summary Report")
            summary = get_data_summary(df)
            
            # Create summary DataFrame
            summary_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Date'],
                'Value': [summary['total_rows'], summary['total_columns'], selected_date]
            }
            
            if 'price_stats' in summary:
                summary_data['Metric'].extend(['Min Price', 'Max Price', 'Avg Price', 'Median Price'])
                summary_data['Value'].extend([
                    f"₹{summary['price_stats']['min_price']:.2f}",
                    f"₹{summary['price_stats']['max_price']:.2f}",
                    f"₹{summary['price_stats']['avg_price']:.2f}",
                    f"₹{summary['price_stats']['median_price']:.2f}"
                ])
            
            if 'rating_stats' in summary:
                summary_data['Metric'].extend(['Min Rating', 'Max Rating', 'Avg Rating', 'Median Rating'])
                summary_data['Value'].extend([
                    f"{summary['rating_stats']['min_rating']:.2f}",
                    f"{summary['rating_stats']['max_rating']:.2f}",
                    f"{summary['rating_stats']['avg_rating']:.2f}",
                    f"{summary['rating_stats']['median_rating']:.2f}"
                ])
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download summary
            summary_csv_link = create_download_link(summary_df, f"summary_{selected_date}", "csv")
            st.markdown(summary_csv_link, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to Daily Data Analysis Dashboard
        
        ### Features:
        - 📤 **Daily File Upload**: Upload CSV/Excel files with automatic date organization
        - 📅 **Date-wise Analysis**: Filter and analyze data by specific dates
        - 📊 **Comprehensive Summary**: View detailed statistics and metrics
        - 📈 **Visual Analysis**: Interactive charts and graphs
        - 🔍 **Search & Filter**: Find specific data with advanced filtering
        - 📥 **Download Options**: Export filtered data and summary reports
        
        ### How to Use:
        1. **Upload Data**: Use the sidebar to upload your daily CSV/Excel files
        2. **Select Date**: Choose a date from the dropdown to analyze that day's data
        3. **Apply Filters**: Use brand, rating, or other filters to narrow down data
        4. **Analyze**: View summary, charts, and detailed data tables
        5. **Download**: Export your filtered data and analysis reports
        
        ### Supported Data Types:
        - Product data with brands, prices, ratings
        - Sales data with dates and amounts
        - Any CSV/Excel file with structured data
        
        **Get started by uploading your first data file using the sidebar! 📁**
        """)

if __name__ == "__main__":
    main()