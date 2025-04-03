import streamlit as st
import plotly.express as px
import pandas as pd
from .BasePage import BasePage
from utils import (
    plot_region_prices,
    plot_property_type_prices
)

class RegionalAnalysisPage(BasePage):
    """Page for analyzing regional property data."""
    
    def __init__(self, df):
        super().__init__(title="Regional Analysis")
        self.df = df
    
    def render(self):
        """Render the regional analysis page."""
        st.header("🗺️ Regional Analysis")
        
        # Determine current theme
        theme = 'light'
        if 'theme' in st.session_state:
            theme = st.session_state.theme
            
        # Define theme-specific colors
        bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
        text_color = '#333333' if theme == 'light' else '#E9EEF6'
        grid_color = '#EEEEEE' if theme == 'light' else '#2A3A4A'
        color_scale = px.colors.sequential.Blues if theme == 'light' else px.colors.sequential.Blues_r
        
        # Custom CSS for better visualization layout
        st.markdown("""
        <style>
        /* Better spacing for charts */
        .element-container {
            margin-bottom: 2rem !important;
        }
        
        /* Chart container styling */
        .stPlotlyChart {
            height: auto !important;
            min-height: 450px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        
        /* Section headers styling */
        h3 {
            margin-top: 2.5rem !important;
            margin-bottom: 1.5rem !important;
            font-size: 1.8rem !important;
        }
        
        /* Better spacing between sections */
        .section-spacing {
            margin-top: 3rem;
            margin-bottom: 1rem;
        }
        
        /* Enhanced select box styling */
        div[data-baseweb="select"] {
            margin-bottom: 1.5rem;
        }
        
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid #4A7CCF !important;
            font-size: 16px !important;
            padding: 12px !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-baseweb="select"] > div:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
            transform: translateY(-2px);
        }
        
        /* Fix for dropdown options appearance */
        div[data-baseweb="select"] span {
            color: currentColor !important;
            font-weight: 500 !important;
        }
        
        /* Fix selected option text */
        div[data-baseweb="select"] [data-testid="stMarkdown"] p {
            color: currentColor !important;
            font-weight: 500 !important;
        }
        
        /* Style for dropdown menu options */
        ul[role="listbox"] li {
            padding: 10px 16px !important;
            font-size: 15px !important;
            transition: background-color 0.2s ease !important;
        }
        
        ul[role="listbox"] li:hover {
            background-color: #F0F5FF !important;
        }
        
        /* Fix dropdown menu in dark theme */
        .stApp[data-theme="dark"] ul[role="listbox"] li:hover {
            background-color: #2A3A4A !important;
        }
        
        .stApp[data-theme="dark"] ul[role="listbox"] {
            background-color: #1E2A3E !important;
            border-color: #2A3A4A !important;
        }
        
        .stApp[data-theme="dark"] ul[role="listbox"] li {
            color: #E9EEF6 !important;
        }
        
        /* Fix for selected option text in dropdown in both themes */
        [data-testid="stSelectbox"] {
            color: inherit !important;
        }
        
        /* Make placeholder more visible */
        [data-baseweb="select"] [aria-selected="true"] {
            color: #4A7CCF !important;
            font-weight: 600 !important;
        }
        
        /* Region analysis card */
        .region-card {
            background-color: var(--bg-light);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0 30px 0;
            border-left: 5px solid #4A7CCF;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Card title styling */
        .card-title {
            color: #4A7CCF;
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(74, 124, 207, 0.2);
            padding-bottom: 10px;
        }
        
        /* Metric value styling */
        .stMetric [data-testid="stMetricValue"] {
            font-size: 24px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Interactive map (commented out due to coordinate issues)
        # st.subheader("Property Prices Across Tunisia")
        # map_fig = plot_interactive_map(self.df)
        # st.plotly_chart(map_fig, use_container_width=True)
        
        # Regional price comparison
        st.markdown("<h3>Regional Price Comparison</h3>", unsafe_allow_html=True)
        
        # Show region-based charts instead
        region_fig = plot_region_prices(self.df)
        st.plotly_chart(region_fig, use_container_width=True, key="region_prices_chart")
        
        # Better spacing between charts
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        # Property type breakdown
        type_fig = plot_property_type_prices(self.df)
        st.plotly_chart(type_fig, use_container_width=True, key="property_type_prices_chart")
        
        # Add a visual spacer
        st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
        
        # Region filter analysis
        st.markdown("<h3>Analysis by Region</h3>", unsafe_allow_html=True)
        
        # Add descriptive text
        st.markdown(
            """
            <div style="background-color: #F8F9FA; padding: 15px; border-radius: 10px; margin-bottom: 20px; 
                border-left: 4px solid #4A7CCF;">
                <p style="margin: 0; color: #555;">
                    Select a region from the dropdown below to view detailed property statistics and price analysis for that specific area.
                    This will help you understand market trends and price distributions in your region of interest.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Create two columns with better proportions
        select_col, info_col = st.columns([1, 2])
        
        with select_col:
            # Region selection with enhanced styling and default message
            st.markdown(f"""<p style='font-weight: 600; margin-bottom: 8px; color: {text_color};'>Select Region:</p>""", unsafe_allow_html=True)
            
            # Add a container with custom styling for better visibility of the select box
            st.markdown(f"""
            <div style="background-color: {'#F7FAFF' if theme == 'light' else '#253245'}; 
                 border-radius: 10px; padding: 2px; border: 1px solid {'#E0E7FF' if theme == 'light' else '#364659'};">
            </div>
            """, unsafe_allow_html=True)
            
            selected_region = st.selectbox(
                "",  # Empty label since we're using the custom label above
                options=[""] + sorted(self.df['region'].unique()),
                index=0,  # Default to empty/placeholder option
                format_func=lambda x: "Choose a region..." if x == "" else x,  # Custom placeholder text
                key="region_selector"  # Add a key for easier CSS targeting
            )
        
        with info_col:
            # Show a quick regional overview on the right
            if selected_region:
                region_data = self.df[self.df['region'] == selected_region]
                count = len(region_data)
                avg_price = region_data['price'].mean()
                avg_area = region_data['area_sqm'].mean()
                
                # Colorful stats display
                st.markdown(
                    f"""
                    <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 8px;">
                        <div style="background-color: {'#E3F2FD' if theme == 'light' else '#1A3A5F'}; color: {'#1565C0' if theme == 'light' else '#90CAF9'}; padding: 15px; border-radius: 10px; text-align: center; flex: 1;">
                            <div style="font-size: 14px; opacity: 0.8;">Properties</div>
                            <div style="font-size: 22px; font-weight: 600;">{count}</div>
                        </div>
                        <div style="background-color: {'#E8F5E9' if theme == 'light' else '#1B3C2A'}; color: {'#2E7D32' if theme == 'light' else '#81C784'}; padding: 15px; border-radius: 10px; text-align: center; flex: 1;">
                            <div style="font-size: 14px; opacity: 0.8;">Avg. Price</div>
                            <div style="font-size: 22px; font-weight: 600;">{avg_price:,.0f} TND</div>
                        </div>
                        <div style="background-color: {'#FFF3E0' if theme == 'light' else '#3E2E1E'}; color: {'#E65100' if theme == 'light' else '#FFB74D'}; padding: 15px; border-radius: 10px; text-align: center; flex: 1;">
                            <div style="font-size: 14px; opacity: 0.8;">Avg. Area</div>
                            <div style="font-size: 22px; font-weight: 600;">{avg_area:.1f} m²</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Continue only if a region is selected
        if not selected_region:
            # Show a message to prompt selection with a nice card design
            st.markdown(
                f"""
                <div style="background-color: {'#F5F5F5' if theme == 'light' else '#253245'}; 
                     border-radius: 10px; padding: 30px; text-align: center; 
                     margin: 40px 0; border: 1px dashed {'#BDBDBD' if theme == 'light' else '#455A64'};">
                    <img src="https://cdn-icons-png.flaticon.com/512/1602/1602624.png" width="80">
                    <h3 style="margin-top: 20px; color: {'#616161' if theme == 'light' else '#B0BEC5'}; font-weight: 500;">Please Select a Region</h3>
                    <p style="color: {'#757575' if theme == 'light' else '#90A4AE'};">Choose a region from the dropdown above to view detailed analysis</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Filter data based on selection
            filtered_df = self.df[self.df['region'] == selected_region]
            
            # Add a title for the region analysis with a badge
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin: 30px 0 20px 0;">
                    <h2 style="margin: 0; color: #1A4A94;">{selected_region} Analysis</h2>
                    <span style="background-color: #4A7CCF; color: white; border-radius: 20px; padding: 5px 15px; 
                           font-size: 14px; margin-left: 15px;">
                        {len(filtered_df)} properties
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Create columns for visualization with better layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Property type distribution in region
                st.markdown(
                    f"""<div class="card-title">Property Types in {selected_region}</div>""", 
                    unsafe_allow_html=True
                )
                
                # Get value counts
                type_counts = filtered_df['property_type'].value_counts()
                
                # Create a theme-aware bar chart
                fig = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    color=type_counts.values,
                    color_continuous_scale=color_scale,
                    labels={"x": "Property Type", "y": "Count"},
                    height=450  # Explicit height
                )
                
                # Improve styling
                fig.update_layout(
                    xaxis_title="Property Type",
                    yaxis_title="Count",
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=text_color, size=14),
                    margin=dict(l=40, r=40, t=20, b=40)
                )
                
                # Add value labels on top of bars
                fig.update_traces(
                    texttemplate='%{y}',
                    textposition='outside',
                    textfont=dict(color=text_color, size=12)
                )
                
                # Grid styling
                fig.update_xaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
                fig.update_yaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True, key="region_property_type_count")
                
                # Add a pie chart for better visualization
                st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
                
                # Create and style pie chart
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title=f"Property Type Distribution in {selected_region}",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    height=350
                )
                
                fig.update_layout(
                    paper_bgcolor=bg_color,
                    font=dict(color=text_color, size=14),
                    legend=dict(
                        font=dict(size=12),
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="region_property_type_pie")
            
            with col2:
                # Price range in region
                st.markdown(
                    f"""<div class="card-title">Price Analysis in {selected_region}</div>""", 
                    unsafe_allow_html=True
                )
                
                # Create a metrics card with a nicer layout
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; 
                         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
                    """,
                    unsafe_allow_html=True
                )
                
                # Use metric tiles with consistent formatting in 2x2 grid
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Min Price", f"{filtered_df['price'].min():,.0f} TND")
                    st.metric("Average Price", f"{filtered_df['price'].mean():,.0f} TND")
                
                with metrics_col2:
                    st.metric("Max Price", f"{filtered_df['price'].max():,.0f} TND")
                    st.metric("Median Price", f"{filtered_df['price'].median():,.0f} TND")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add price distribution chart with improved styling
                fig = px.histogram(
                    filtered_df,
                    x="price",
                    nbins=20,
                    color_discrete_sequence=["#4A7CCF"],
                    title=f"Price Distribution in {selected_region}",
                    labels={"price": "Price (TND)"},
                    height=400  # Increased height
                )
                
                # Layout styling
                fig.update_layout(
                    xaxis_tickformat=',.0f',
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=text_color, size=13),
                    title=dict(font=dict(size=16)),
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                
                # Grid styling
                fig.update_xaxes(showgrid=True, gridcolor=grid_color)
                fig.update_yaxes(showgrid=True, gridcolor=grid_color)
                
                # Display the histogram
                st.plotly_chart(fig, use_container_width=True, key="region_price_histogram")
            
            # Add another section with additional regional analysis in a full-width layout
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            
            # Create tabbed analysis for more detailed insights
            tabs = st.tabs(["Price by Bedroom", "Price by Area", "Property Features"])
            
            with tabs[0]:
                # Create price by bedroom analysis for selected region
                bedroom_data = filtered_df.groupby('bedrooms')['price'].mean().reset_index()
                if not bedroom_data.empty:
                    fig = px.bar(
                        bedroom_data,
                        x='bedrooms',
                        y='price',
                        color='price',
                        color_continuous_scale=color_scale,
                        labels={"price": "Average Price (TND)", "bedrooms": "Number of Bedrooms"},
                        height=450  # Explicit height
                    )
                    
                    # Styling
                    fig.update_layout(
                        title=f"Average Price by Bedroom Count in {selected_region}",
                        xaxis={'categoryorder':'array', 'categoryarray':sorted(filtered_df['bedrooms'].unique())},
                        yaxis_tickformat=',.0f',
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color,
                        font=dict(color=text_color, size=14),
                        margin=dict(l=50, r=50, t=60, b=40)
                    )
                    
                    # Add value labels
                    fig.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        textfont=dict(color=text_color, size=12)
                    )
                    
                    # Grid styling
                    fig.update_xaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
                    fig.update_yaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, key="region_price_by_bedroom")
                else:
                    st.info(f"No bedroom data available for {selected_region}")
                    
            with tabs[1]:
                # Add area analysis with scatter plot
                if not filtered_df.empty:
                    fig = px.scatter(
                        filtered_df,
                        x='area_sqm',
                        y='price',
                        color='property_type',
                        size='price',
                        size_max=25,
                        hover_data=['bedrooms', 'bathrooms', 'property_age'],
                        title=f"Price vs Area in {selected_region}",
                        labels={"area_sqm": "Area (m²)", "price": "Price (TND)"},
                        height=450
                    )
                    
                    # Add trendline
                    fig.update_layout(
                        yaxis_tickformat=',.0f',
                        paper_bgcolor=bg_color,
                        plot_bgcolor=bg_color,
                        font=dict(color=text_color, size=14),
                        margin=dict(l=50, r=50, t=60, b=40),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.15,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    
                    fig.update_xaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
                    fig.update_yaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
                    
                    st.plotly_chart(fig, use_container_width=True, key="region_price_vs_area")
                else:
                    st.info(f"No area data available for {selected_region}")
                    
            with tabs[2]:
                # Property features analysis
                if not filtered_df.empty:
                    # Create 3 columns for amenities analysis
                    cols = st.columns(3)
                    
                    # Amenities to analyze
                    amenities = ['has_elevator', 'has_garden', 'has_parking']
                    amenity_names = ['Elevator', 'Garden', 'Parking']
                    
                    for i, (feature, name) in enumerate(zip(amenities, amenity_names)):
                        with cols[i]:
                            # Calculate percentages
                            has_feature = filtered_df[feature].sum()
                            total = len(filtered_df)
                            percentage = (has_feature / total) * 100 if total > 0 else 0
                            
                            # Create a gauge chart
                            fig = px.pie(
                                values=[percentage, 100-percentage],
                                names=['Has ' + name, 'No ' + name],
                                hole=0.7,
                                color_discrete_sequence=['#4A7CCF', '#E0E0E0']
                            )
                            
                            fig.update_layout(
                                title=f"{name} Availability",
                                annotations=[dict(text=f'{percentage:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                                showlegend=False,
                                height=250,
                                paper_bgcolor=bg_color,
                                margin=dict(l=10, r=10, t=40, b=10)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"region_{name.lower()}_availability")
                            
                            # Add average price comparison
                            avg_with = filtered_df[filtered_df[feature] == True]['price'].mean()
                            avg_without = filtered_df[filtered_df[feature] == False]['price'].mean()
                            
                            if not pd.isna(avg_with) and not pd.isna(avg_without) and avg_without != 0:
                                diff_pct = ((avg_with / avg_without) - 1) * 100
                                
                                if diff_pct > 0:
                                    st.markdown(f"Properties with {name.lower()} are **{diff_pct:.1f}%** more expensive")
                                else:
                                    st.markdown(f"Properties with {name.lower()} are **{abs(diff_pct):.1f}%** less expensive")
                else:
                    st.info(f"No feature data available for {selected_region}") 