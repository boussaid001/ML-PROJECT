import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from .BasePage import BasePage
from utils import (
    plot_price_vs_area,
    plot_correlation_heatmap,
    format_price,
    display_matplotlib_fig,
    apply_custom_theme
)
from database import save_user_search

class ExploreMarketPage(BasePage):
    """Page for exploring the real estate market."""
    
    def __init__(self, df):
        super().__init__(title="Explore Market")
        self.df = df
    
    def render(self):
        """Render the market exploration page."""
        st.header("📊 Tunisia Real Estate Market Explorer")
        
        # Apply theme-aware styling
        apply_custom_theme()
        
        # Get theme colors
        theme = 'light'
        if 'theme' in st.session_state:
            theme = st.session_state.theme
            
        # Define theme colors
        bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
        text_color = '#333333' if theme == 'light' else '#E9EEF6'
        grid_color = '#E9EEF6' if theme == 'dark' else '#333333'
        color_scale = ['#4A7CCF', '#1A4A94', '#F8B400', '#FC5C65'] if theme == 'light' else ['#5D8FE2', '#4A7CCF', '#F8B400', '#FC5C65']
        
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
        }
        
        /* Section headers styling */
        h3 {
            margin-top: 2rem !important;
            margin-bottom: 1.5rem !important;
            font-size: 1.8rem !important;
        }
        
        /* Property counter styling */
        .property-counter {
            background-color: var(--bg-light);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 10px var(--shadow);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .count-number {
            font-size: 32px;
            font-weight: 700;
            color: var(--primary);
        }
        
        .count-label {
            font-size: 18px;
            color: var(--text-medium);
            margin-left: 10px;
        }
        
        /* Filter section styling */
        .filter-section {
            background-color: var(--bg-light);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 10px var(--shadow);
            margin-bottom: 30px;
        }
        
        /* Chart container styling */
        .chart-container {
            background-color: var(--bg-light);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 10px var(--shadow);
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Data overview
        with st.expander("Dataset Preview"):
            st.dataframe(self.df.head(10), height=400)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Shape:", self.df.shape)
            with col2:
                st.write("Missing Values:", self.df.isnull().sum().sum())
        
        # Display property counter
        property_count = len(self.df)
        
        # Property counter card
        st.markdown(f"""
        <div class="property-counter">
            <div style="display: flex; align-items: center;">
                <span class="count-number">{property_count:,}</span>
                <span class="count-label">Properties Available</span>
            </div>
            <div style="font-size: 40px; color: var(--primary);">🏠</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Filters section
        st.markdown("<h3>Filter Properties</h3>", unsafe_allow_html=True)
        
        # Set up filter columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""<p style='font-weight: 600; margin-bottom: 8px; color: {text_color};'>Property Type:</p>""", unsafe_allow_html=True)
            property_type = st.selectbox(
                "",
                options=["All"] + sorted(self.df['property_type'].unique().tolist()),
                key="property_type_select"
            )
        
        with col2:
            st.markdown(f"""<p style='font-weight: 600; margin-bottom: 8px; color: {text_color};'>Region:</p>""", unsafe_allow_html=True)
            region = st.selectbox(
                "",
                options=["All"] + sorted(self.df['region'].unique().tolist()),
                key="region_select"
            )
        
        with col3:
            st.markdown(f"""<p style='font-weight: 600; margin-bottom: 8px; color: {text_color};'>Price Range:</p>""", unsafe_allow_html=True)
            min_price = st.number_input("Min Price (TND)", min_value=0, max_value=5000000, value=0, step=50000)
            max_price = st.number_input("Max Price (TND)", min_value=0, max_value=5000000, value=5000000, step=50000)
        
        # Additional filters in an expander
        with st.expander("More Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                min_area = st.number_input("Min Area (sqm)", min_value=0, max_value=1000, value=0, step=10)
                min_bedrooms = st.number_input("Min Bedrooms", min_value=0, max_value=10, value=0, step=1)
            
            with col2:
                max_area = st.number_input("Max Area (sqm)", min_value=0, max_value=1000, value=1000, step=10)
                max_bathrooms = st.number_input("Min Bathrooms", min_value=0, max_value=10, value=0, step=1)
        
        # Apply filters button
        if st.button("Apply Filters", key="apply_filters"):
            # Save search to database if user is logged in
            if st.session_state.get('logged_in', False) and st.session_state.get('user_id'):
                search_term = f"Property: {property_type if property_type != 'All' else 'Any'}, Region: {region if region != 'All' else 'Any'}"
                save_user_search(
                    st.session_state.user_id,
                    search_term,
                    region if region != 'All' else None,
                    property_type if property_type != 'All' else None,
                    min_price if min_price > 0 else None,
                    max_price if max_price < 5000000 else None,
                    min_bedrooms if min_bedrooms > 0 else None
                )
        
        # Filter the dataframe based on selections
        filtered_df = self.df.copy()
        
        # Apply property type filter
        if property_type != "All":
            filtered_df = filtered_df[filtered_df['property_type'] == property_type]
        
        # Apply region filter
        if region != "All":
            filtered_df = filtered_df[filtered_df['region'] == region]
        
        # Apply price range filter
        filtered_df = filtered_df[
            (filtered_df['price'] >= min_price) & 
            (filtered_df['price'] <= max_price)
        ]
        
        # Apply area filter
        filtered_df = filtered_df[
            (filtered_df['area_sqm'] >= min_area) & 
            (filtered_df['area_sqm'] <= max_area)
        ]
        
        # Apply bedroom filter
        if min_bedrooms > 0:
            filtered_df = filtered_df[filtered_df['bedrooms'] >= min_bedrooms]
        
        # Apply bathroom filter
        if max_bathrooms > 0:
            filtered_df = filtered_df[filtered_df['bathrooms'] >= max_bathrooms]
        
        # Show count of filtered properties
        st.markdown(f"<h4>Showing {len(filtered_df):,} properties</h4>", unsafe_allow_html=True)
        
        # Display the filtered properties in a dataframe
        if not filtered_df.empty:
            # Format dataframe for display
            display_df = filtered_df[['property_type', 'region', 'area_sqm', 'bedrooms', 'bathrooms', 'price']].copy()
            display_df.columns = ['Property Type', 'Region', 'Area (sqm)', 'Bedrooms', 'Bathrooms', 'Price (TND)']
            display_df = display_df.sort_values('Price (TND)', ascending=True)
            
            # Display dataframe
            st.dataframe(display_df, use_container_width=True)
            
            # Property Analysis Section
            st.markdown("<h3>Property Analysis</h3>", unsafe_allow_html=True)
            
            # Price vs Area scatter plot
            st.subheader("Price vs Area")
            price_area_fig = plot_price_vs_area(filtered_df)
            st.plotly_chart(price_area_fig, use_container_width=True, key="filtered_price_area_chart")
            
            # Feature correlation heatmap
            st.subheader("Feature Correlation")
            corr_fig = plot_correlation_heatmap(filtered_df)
            display_matplotlib_fig(corr_fig)
            
            # Property distribution by type
            st.subheader("Property Type Distribution")
            type_counts = filtered_df['property_type'].value_counts()
            
            # Get colors based on current theme
            colors = ['#4A7CCF', '#1A4A94', '#F8B400', '#FC5C65'] if theme == 'light' else ['#5D8FE2', '#4A7CCF', '#F8B400', '#FC5C65']
            
            # Create figure with theme-aware styling
            fig, ax = plt.subplots(figsize=(10, 6))
            # Set figure properties after creation
            fig.patch.set_facecolor('#FFFFFF' if theme == 'light' else '#1E2A3E')
            ax.set_facecolor('#FFFFFF' if theme == 'light' else '#1E2A3E')
            
            # Create bar chart
            bars = ax.bar(type_counts.index, type_counts.values, color=colors[:len(type_counts)])
            
            # Add labels and styling
            ax.set_xlabel('Property Type', color=text_color)
            ax.set_ylabel('Count', color=text_color)
            ax.set_title('Property Type Distribution', color=text_color, fontsize=16)
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_color('#E9EEF6' if theme == 'dark' else '#333333')
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}',
                        ha='center', va='bottom', color=text_color)
            
            # Display the figure
            display_matplotlib_fig(fig)
            
            # Price distribution by property type
            st.subheader("Price Distribution by Property Type")
            
            # Create figure with theme-aware styling
            fig, ax = plt.subplots(figsize=(12, 8))
            # Set figure properties after creation
            fig.patch.set_facecolor('#FFFFFF' if theme == 'light' else '#1E2A3E')
            ax.set_facecolor('#FFFFFF' if theme == 'light' else '#1E2A3E')
            
            # Create boxplot
            boxplot = ax.boxplot([filtered_df[filtered_df['property_type'] == t]['price'] for t in filtered_df['property_type'].unique()],
                      labels=filtered_df['property_type'].unique(),
                      patch_artist=True)
            
            # Set colors for boxplots based on theme
            for patch, color in zip(boxplot['boxes'], colors[:len(filtered_df['property_type'].unique())]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Style whiskers, caps, and medians
            for whisker in boxplot['whiskers']:
                whisker.set(color=text_color, linewidth=1.5)
            for cap in boxplot['caps']:
                cap.set(color=text_color, linewidth=1.5)
            for median in boxplot['medians']:
                median.set(color='white', linewidth=2)
            for flier in boxplot['fliers']:
                flier.set(marker='o', markerfacecolor='none', markersize=5, 
                       markeredgecolor=text_color, alpha=0.5)
            
            # Add labels and styling
            ax.set_xlabel('Property Type', color=text_color, fontsize=14)
            ax.set_ylabel('Price (TND)', color=text_color, fontsize=14)
            ax.set_title('Price Distribution by Property Type', color=text_color, fontsize=16)
            ax.tick_params(axis='x', colors=text_color, labelsize=12)
            ax.tick_params(axis='y', colors=text_color, labelsize=12)
            
            # Format y-axis to show comma separators for thousands
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_color('#E9EEF6' if theme == 'dark' else '#333333')
            
            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7, color='#DDDDDD' if theme == 'light' else '#2A3A4A')
            
            # Display the figure
            display_matplotlib_fig(fig)
            
        else:
            st.warning("No properties match your filter criteria. Please adjust your filters.")
        
        # Price distribution
        st.subheader("Property Price Distribution")
        
        fig = px.histogram(
            self.df, 
            x="price", 
            nbins=50,
            color_discrete_sequence=["#4A7CCF"],
            title="Distribution of Property Prices in Tunisia",
            labels={"price": "Price (TND)"},
            height=500  # Set explicit height
        )
        fig.update_layout(
            xaxis_tickformat=',.0f',
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color, size=14),
            title=dict(font=dict(size=20)),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        fig.update_xaxes(
            showgrid=True, 
            gridcolor=grid_color, 
            title_font=dict(size=14)
        )
        fig.update_yaxes(
            showgrid=True, 
            gridcolor=grid_color, 
            title_font=dict(size=14)
        )
        
        # Show chart with full width
        st.plotly_chart(fig, use_container_width=True, key="price_distribution_chart")
        
        # Price statistics
        st.markdown("<h4 style='margin-top: 1.5rem; margin-bottom: 1rem;'>Price Statistics</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Price", format_price(self.df['price'].mean()))
        with col2:
            st.metric("Median Price", format_price(self.df['price'].median()))
        with col3:
            st.metric("Minimum Price", format_price(self.df['price'].min()))
        with col4:
            st.metric("Maximum Price", format_price(self.df['price'].max()))
        
        # Feature relationships
        st.subheader("Property Features Analysis")
        
        # Price vs Area - Using larger charts
        price_vs_area_fig = plot_price_vs_area(self.df)
        st.plotly_chart(price_vs_area_fig, use_container_width=True, key="price_vs_area_chart")
        
        # Using a container for better spacing
        st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
        
        # Other feature relationships
        feature_cols = st.columns(2)
        
        with feature_cols[0]:
            # Price vs bedrooms
            bedroom_data = self.df.groupby('bedrooms')['price'].mean().reset_index()
            fig = px.bar(
                bedroom_data,
                x='bedrooms',
                y='price',
                title="Average Price by Number of Bedrooms",
                labels={"price": "Average Price (TND)", "bedrooms": "Number of Bedrooms"},
                color='price',
                color_continuous_scale=color_scale,
                height=450  # Set explicit height
            )
            fig.update_layout(
                xaxis={'categoryorder':'array', 'categoryarray':sorted(self.df['bedrooms'].unique())},
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=text_color, size=14),
                title=dict(font=dict(size=18)),
                margin=dict(l=40, r=40, t=80, b=40)
            )
            # Add value labels on top of bars
            fig.update_traces(
                texttemplate='%{y:,.0f}',
                textposition='outside',
                textfont=dict(color=text_color, size=11)
            )
            fig.update_xaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=13))
            fig.update_yaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=13))
            
            st.plotly_chart(fig, use_container_width=True, key="bedrooms_price_chart")
        
        with feature_cols[1]:
            # Price vs property age
            fig = px.scatter(
                self.df,
                x='property_age',
                y='price',
                color='property_type',
                opacity=0.7,
                title="Price vs Property Age",
                labels={"price": "Price (TND)", "property_age": "Property Age (years)"},
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=450  # Set explicit height
            )
            fig.update_layout(
                yaxis_tickformat=',.0f',
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color,
                font=dict(color=text_color, size=14),
                title=dict(font=dict(size=18)),
                margin=dict(l=40, r=40, t=80, b=40)
            )
            fig.update_xaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=13))
            fig.update_yaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=13))
            
            # Update legend
            fig.update_layout(
                legend=dict(
                    bgcolor=bg_color,
                    font=dict(color=text_color, size=12),
                    title=dict(text="Property Type", font=dict(size=13)),
                    borderwidth=1,
                    bordercolor=grid_color
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, key="property_age_price_chart")
        
        # Better spacing between sections
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        
        # Get correlation heatmap from utils
        corr_fig = plot_correlation_heatmap(self.df)
        
        # Use the helper function for better display
        display_matplotlib_fig(corr_fig)
        
        # Better spacing between sections
        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        
        # Impact of boolean features
        st.subheader("Impact of Property Amenities")
        
        # Using 3 columns with better spacing
        amenity_cols = st.columns(3)
        
        bool_features = ['has_elevator', 'has_garden', 'has_parking']
        for i, feature in enumerate(bool_features):
            with amenity_cols[i]:
                avg_with = self.df[self.df[feature] == True]['price'].mean()
                avg_without = self.df[self.df[feature] == False]['price'].mean()
                diff_pct = (avg_with / avg_without - 1) * 100
                
                feature_name = feature.replace('has_', '').capitalize()
                
                fig = px.bar(
                    x=['With ' + feature_name, 'Without ' + feature_name],
                    y=[avg_with, avg_without],
                    color=[avg_with, avg_without],
                    color_continuous_scale=color_scale,
                    title=f"Price With/Without {feature_name}",
                    labels={"x": "", "y": "Average Price (TND)"},
                    height=400  # Adjusted height
                )
                fig.update_layout(
                    yaxis_tickformat=',.0f',
                    paper_bgcolor=bg_color,
                    plot_bgcolor=bg_color,
                    font=dict(color=text_color, size=14),
                    title=dict(font=dict(size=16)),
                    margin=dict(l=30, r=30, t=70, b=40)
                )
                # Add value labels on bars
                fig.update_traces(
                    texttemplate='%{y:,.0f}',
                    textposition='outside',
                    textfont=dict(color=text_color, size=11)
                )
                fig.update_xaxes(showgrid=True, gridcolor=grid_color)
                fig.update_yaxes(showgrid=True, gridcolor=grid_color)
                
                st.plotly_chart(fig, use_container_width=True, key=f"amenity_{feature}_chart")
                
                # Format the percentage with a nice color
                if diff_pct > 0:
                    color = "#28a745"  # Green for positive
                else:
                    color = "#dc3545"  # Red for negative
                    
                st.markdown(
                    f"Properties with {feature_name.lower()} are "
                    f"<span style='color:{color}; font-weight:bold;'>{diff_pct:.1f}%</span> "
                    f"{'more' if diff_pct > 0 else 'less'} expensive on average.", 
                    unsafe_allow_html=True
                ) 