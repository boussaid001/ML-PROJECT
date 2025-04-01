import streamlit as st
import plotly.express as px
from src.pages.base_page import BasePage
from src.utils import (
    plot_interactive_map,
    plot_region_prices,
    plot_property_type_prices,
    plot_price_vs_area,
    format_price
)

class HomePage(BasePage):
    """Home page implementation"""
    def render(self):
        st.header("🇹🇳 Tunisia Real Estate Market Analysis")
        self._render_feature_cards()
        self._render_dataset_overview()
        self._render_map()
        self._render_market_overview()
        self._render_price_trends()

    def _render_feature_cards(self):
        st.markdown("""
        <style>
        .feature-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }
        .feature-card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            padding: 20px;
            flex: 1;
            min-width: 200px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-top: 4px solid #E41E25;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }
        .feature-card h3 {
            margin-top: 0;
            color: #E41E25;
        }
        .feature-card p {
            color: #333333;
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
            color: #E41E25;
        }
        </style>
        
        <div class="feature-container">
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>Market Analysis</h3>
                <p>Explore Tunisia's real estate market with interactive visualizations and statistics.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🗺️</div>
                <h3>Regional Insights</h3>
                <p>Compare property prices across all 24 governorates in Tunisia.</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🏘️</div>
                <h3>Property Valuation</h3>
                <p>Get accurate price predictions based on location, property features, and market trends.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_dataset_overview(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Properties", f"{self.df.shape[0]:,}")
        with col2:
            st.metric("Regions", f"{len(self.df['region'].unique())}")
        with col3:
            st.metric("Property Types", f"{len(self.df['property_type'].unique())}")

    def _render_map(self):
        st.subheader("🗺️ Property Prices Across Tunisia")
        map_fig = plot_interactive_map(self.df)
        st.plotly_chart(map_fig, use_container_width=True)

    def _render_market_overview(self):
        st.subheader("📊 Market Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_regional_price_range()
        with col2:
            self._render_property_type_insights()

    def _render_regional_price_range(self):
        region_prices = self.df.groupby('region')['price'].mean().sort_values()
        cheapest_region = region_prices.index[0]
        cheapest_price = region_prices.iloc[0]
        most_expensive_region = region_prices.index[-1]
        most_expensive_price = region_prices.iloc[-1]
        
        st.markdown(f"""
        <div class="card">
            <h3>Regional Price Range</h3>
            <p>Most expensive region: <strong>{most_expensive_region}</strong> with average price of <strong>{format_price(most_expensive_price)}</strong></p>
            <p>Most affordable region: <strong>{cheapest_region}</strong> with average price of <strong>{format_price(cheapest_price)}</strong></p>
            <p>Price difference: <strong>{format_price(most_expensive_price - cheapest_price)}</strong> ({(most_expensive_price/cheapest_price - 1)*100:.1f}% higher)</p>
        </div>
        """, unsafe_allow_html=True)

    def _render_property_type_insights(self):
        type_prices = self.df.groupby('property_type')['price'].mean().sort_values()
        cheapest_type = type_prices.index[0]
        most_expensive_type = type_prices.index[-1]
        
        st.markdown(f"""
        <div class="card">
            <h3>Property Type Insights</h3>
            <p>Most expensive property type: <strong>{most_expensive_type}</strong> with average price of <strong>{format_price(type_prices[most_expensive_type])}</strong></p>
            <p>Most affordable property type: <strong>{cheapest_type}</strong> with average price of <strong>{format_price(type_prices[cheapest_type])}</strong></p>
            <p>Price ratio: <strong>{type_prices[most_expensive_type]/type_prices[cheapest_type]:.1f}x</strong> difference between highest and lowest</p>
        </div>
        """, unsafe_allow_html=True)

    def _render_price_trends(self):
        st.subheader("📈 Price Trends")
        st.plotly_chart(plot_property_type_prices(self.df), use_container_width=True) 