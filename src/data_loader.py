import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CovidDataLoader:
    def __init__(self, local_file='owid-covid-data.csv'):
        self.local_file = local_file
        self.remote_url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        self.df = None
        self.countries_of_interest = [
            'United States', 'India', 'Brazil', 'United Kingdom', 
            'Kenya', 'South Africa', 'Germany'
        ]
        self.key_columns = [
            'total_cases', 'new_cases', 'total_deaths', 
            'new_deaths', 'total_vaccinations', 'people_vaccinated'
        ]
        
        # Set visualization styles
        plt.style.use('ggplot')
        sns.set_palette("husl")
    
    def load_data(self):
        """Load COVID-19 data from local file or download from web."""
        try:
            if Path(self.local_file).exists():
                logger.info("Loading data from local file...")
                self.df = pd.read_csv(self.local_file)
            else:
                logger.info("Downloading data from web...")
                self.df = pd.read_csv(self.remote_url)
                self.df.to_csv(self.local_file, index=False)
                
            logger.info(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_basic_info(self):
        """Return basic information about the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'head': self.df.head(),
            'data_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            }
        }
        return info

    def clean_data(self):
        """Clean and preprocess the COVID-19 data."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Starting data cleaning process...")
        
        # Convert date column
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Filter countries
        self.df_clean = self.df[self.df['location'].isin(self.countries_of_interest)].copy()
        
        # Handle missing values
        logger.info("Missing values before cleaning:")
        logger.info(self.df_clean[self.key_columns].isnull().sum())
        
        # Forward fill missing values within each country
        self.df_clean[self.key_columns] = self.df_clean.groupby('location')[self.key_columns].fillna(method='ffill')
        
        # Calculate derived metrics
        self.df_clean['death_rate'] = self.df_clean['total_deaths'] / self.df_clean['total_cases']
        self.df_clean['cases_per_million'] = self.df_clean['total_cases'] / (self.df_clean['population'] / 1e6)
        self.df_clean['deaths_per_million'] = self.df_clean['total_deaths'] / (self.df_clean['population'] / 1e6)
        
        # Calculate vaccination percentage
        self.df_clean['pct_vaccinated'] = (self.df_clean['people_vaccinated'] / 
                                          self.df_clean['population']) * 100
        
        logger.info("Data cleaning completed")
        return self.df_clean
    
    def get_latest_data(self):
        """Get the latest data for each country."""
        if not hasattr(self, 'df_clean'):
            raise ValueError("Data not cleaned. Call clean_data() first.")
            
        latest_date = self.df_clean['date'].max()
        return self.df_clean[self.df_clean['date'] == latest_date]
    
    def plot_total_cases(self):
        """Plot total cases over time for selected countries."""
        plt.figure(figsize=(14, 8))
        for country in self.countries_of_interest:
            country_data = self.df_clean[self.df_clean['location'] == country]
            plt.plot(country_data['date'], country_data['total_cases'], label=country)
        
        plt.title('Total COVID-19 Cases Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Total Cases', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_new_cases_trend(self, window=7):
        """Plot new cases with rolling average."""
        plt.figure(figsize=(14, 8))
        for country in self.countries_of_interest:
            country_data = self.df_clean[self.df_clean['location'] == country]
            country_data['new_cases_smoothed'] = country_data['new_cases'].rolling(window=window).mean()
            plt.plot(country_data['date'], country_data['new_cases_smoothed'], label=country)
        
        plt.title(f'Daily New COVID-19 Cases ({window}-day average)', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('New Cases', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_total_deaths(self):
        """Plot total deaths over time."""
        plt.figure(figsize=(14, 8))
        for country in self.countries_of_interest:
            country_data = self.df_clean[self.df_clean['location'] == country]
            plt.plot(country_data['date'], country_data['total_deaths'], label=country)
        
        plt.title('Total COVID-19 Deaths Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Total Deaths', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_comparison_metrics(self):
        """Plot death rates and cases per million comparisons."""
        latest_data = self.get_latest_data()
        
        # Death rates comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='location', y='death_rate', 
                   data=latest_data.sort_values('death_rate', ascending=False))
        plt.title('COVID-19 Death Rate by Country', fontsize=16)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Death Rate', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        death_rate_fig = plt.gcf()
        
        # Cases per million comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(x='location', y='cases_per_million', 
                   data=latest_data.sort_values('cases_per_million', ascending=False))
        plt.title('COVID-19 Cases per Million People', fontsize=16)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Cases per Million', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        cases_per_million_fig = plt.gcf()
        
        return death_rate_fig, cases_per_million_fig
    
    def plot_vaccination_progress(self):
        """Plot total vaccinations over time for selected countries."""
        plt.figure(figsize=(14, 8))
        for country in self.countries_of_interest:
            country_data = self.df_clean[self.df_clean['location'] == country]
            plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)
        
        plt.title('Total COVID-19 Vaccinations Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Total Vaccinations', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_vaccination_percentage(self):
        """Plot percentage of population vaccinated by country."""
        latest_data = self.get_latest_data()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='location', y='pct_vaccinated',
                   data=latest_data.sort_values('pct_vaccinated', ascending=False))
        plt.title('Percentage of Population Vaccinated', fontsize=16)
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Percentage Vaccinated', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_cases_choropleth(self):
        """Create a world map showing COVID-19 cases per million."""
        latest_global_data = self.df[self.df['date'] == self.df['date'].max()].copy()
        latest_global_data['cases_per_million'] = (
            latest_global_data['total_cases'] / (latest_global_data['population'] / 1e6)
        )
        
        fig = px.choropleth(
            latest_global_data,
            locations="iso_code",
            color="cases_per_million",
            hover_name="location",
            color_continuous_scale=px.colors.sequential.Plasma,
            title=f"COVID-19 Cases per Million People as of {latest_global_data['date'].iloc[0].date()}",
            labels={'cases_per_million': 'Cases per Million'}
        )
        return fig
    
    def plot_vaccination_choropleth(self):
        """Create a world map showing vaccination percentages."""
        latest_global_data = self.df[self.df['date'] == self.df['date'].max()].copy()
        latest_global_data['pct_vaccinated'] = (
            latest_global_data['people_vaccinated'] / latest_global_data['population']
        ) * 100
        
        fig = px.choropleth(
            latest_global_data,
            locations="iso_code",
            color="pct_vaccinated",
            hover_name="location",
            color_continuous_scale=px.colors.sequential.Viridis,
            title=f"Percentage of Population Vaccinated as of {latest_global_data['date'].iloc[0].date()}",
            labels={'pct_vaccinated': '% Vaccinated'}
        )
        return fig

    def get_key_insights(self):
        """Generate key insights from the latest data."""
        if not hasattr(self, 'df_clean'):
            raise ValueError("Data not cleaned. Call clean_data() first.")
            
        latest_data = self.get_latest_data()
        latest_date = latest_data['date'].iloc[0]
        
        # Find countries with highest metrics
        highest_cases = latest_data.loc[latest_data['total_cases'].idxmax()]
        highest_deaths = latest_data.loc[latest_data['total_deaths'].idxmax()]
        highest_vaccination = latest_data.loc[latest_data['pct_vaccinated'].idxmax()]
        
        insights = {
            'date': latest_date.date(),
            'highest_cases': {
                'country': highest_cases['location'],
                'cases': int(highest_cases['total_cases']/1e6)
            },
            'highest_deaths': {
                'country': highest_deaths['location'],
                'deaths': int(highest_deaths['total_deaths']/1e3)
            },
            'highest_vaccination': {
                'country': highest_vaccination['location'],
                'percentage': round(highest_vaccination['pct_vaccinated'], 1)
            },
            'death_rate_range': {
                'max': round(latest_data['death_rate'].max()*100, 1),
                'min': round(latest_data['death_rate'].min()*100, 1)
            }
        }
        
        return insights
    
    def get_case_trends(self):
        """Analyze case trends for each country."""
        if not hasattr(self, 'df_clean'):
            raise ValueError("Data not cleaned. Call clean_data() first.")
            
        trends = {}
        for country in self.countries_of_interest:
            country_data = self.df_clean[self.df_clean['location'] == country]
            peak_idx = country_data['new_cases'].idxmax()
            trends[country] = {
                'peak_date': country_data.loc[peak_idx, 'date'].date(),
                'peak_cases': int(country_data.loc[peak_idx, 'new_cases'])
            }
        
        return trends

    def save_visualizations(self, output_dir='../outputs'):
        """Save all visualizations to files."""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save static plots
        static_plots = [
            ('total_cases', self.plot_total_cases()),
            ('new_cases_trend', self.plot_new_cases_trend()),
            ('total_deaths', self.plot_total_deaths()),
            ('vaccination_progress', self.plot_vaccination_progress()),
            ('vaccination_percentage', self.plot_vaccination_percentage())
        ]
        
        for name, fig in static_plots:
            fig.savefig(f'{output_dir}/{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Save choropleth maps as HTML
        cases_map = self.plot_cases_choropleth()
        vaccination_map = self.plot_vaccination_choropleth()
        cases_map.write_html(f'{output_dir}/cases_map.html')
        vaccination_map.write_html(f'{output_dir}/vaccination_map.html')
        
        logger.info(f"All visualizations saved to {output_dir}")
    
    def export_latest_data(self, output_dir='../outputs'):
        """Export latest COVID-19 data to CSV."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        latest_data = self.get_latest_data()
        output_file = f'{output_dir}/latest_covid_data.csv'
        latest_data.to_csv(output_file, index=False)
        logger.info(f"Latest data exported to {output_file}")

if __name__ == "__main__":
    loader = CovidDataLoader()
    df = loader.load_data()
    df_clean = loader.clean_data()
    latest_data = loader.get_latest_data()
    print("\nLatest date in dataset:", latest_data['date'].iloc[0].date())
    print("\nLatest statistics per country:")
    print(latest_data[['location', 'total_cases', 'total_deaths', 'death_rate']])
    
    # Generate and display plots
    loader.plot_total_cases()
    loader.plot_new_cases_trend()
    loader.plot_total_deaths()
    loader.plot_comparison_metrics()
    loader.plot_vaccination_progress()
    loader.plot_vaccination_percentage()
    
    # Generate and display choropleth maps
    cases_map = loader.plot_cases_choropleth()
    vaccination_map = loader.plot_vaccination_choropleth()
    cases_map.show()
    vaccination_map.show()
    plt.show()
    
    # Display key insights
    insights = loader.get_key_insights()
    print("\nKey Insights:")
    print(f"1. As of {insights['date']}, {insights['highest_cases']['country']} had the highest total cases",
          f"with {insights['highest_cases']['cases']} million cases.")
    print(f"2. {insights['highest_deaths']['country']} had the highest total deaths with",
          f"{insights['highest_deaths']['deaths']} thousand deaths.")
    print(f"3. {insights['highest_vaccination']['country']} had the highest vaccination rate with",
          f"{insights['highest_vaccination']['percentage']}% of its population vaccinated.")
    print(f"4. The global death rate varies significantly by country, with the highest being",
          f"{insights['death_rate_range']['max']}% and the lowest being {insights['death_rate_range']['min']}%.")
    
    # Display case trends
    trends = loader.get_case_trends()
    print("\nCase Trends:")
    for country, data in trends.items():
        print(f"- {country} had its peak daily cases on {data['peak_date']} with {data['peak_cases']} new cases.")
    
    # Save visualizations and data
    loader.save_visualizations()
    loader.export_latest_data()
    print("\nReport components saved successfully!")
