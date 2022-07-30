###########################################################################################################
######## author = Jeffrey Siew
######## insitution = N/A
######## website = https://github.com/jeffrey-siew
######## version = 1.0
######## status = Initial Release
######## deployed at = streamlit cloud
######## layout inspired by https://share.streamlit.io/tdenzl/bulian/main/BuLiAn.py
###########################################################################################################

import s3fs

from matplotlib.backends.backend_agg import RendererAgg
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib
from streamlit_lottie import st_lottie
import requests
import json
import pyarrow.parquet as pq

import boto3
import base64
import uuid
import re

import plotly.express as px

#############################
### LOADING CONFIGURATION ### 
#############################

# EMPTY

################################
### FUNCTION FOR THE WEB APP ###
################################

###
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

###
def download_df(df):
    """
    Download an object from AWS
    Example key: my/key/some_file.txt
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    file_name = 'web_app_data.csv'
    file_type = 'csv'

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{file_name}" id="{button_id}" href="data:file/{file_type};base64,{b64}">Download {file_name}</a><br></br>'
    )
    return dl_link

###
def download_aws_object(bucket, key):
    """
    Download an object from AWS
    Example key: my/key/some_file.txt
    """
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, key)
    file_name = key.split('/')[-1] # e.g. some_file.txt
    file_type = file_name.split('.')[-1] # e.g. txt
    b64 = base64.b64encode(obj.get()['Body'].read()).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{file_name}" id="{button_id}" href="data:file/{file_type};base64,{b64}">Download {file_name}</a><br></br>'
    )
    return dl_link

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def setup_data(filename):
    # Create connection object.
    # `anon=False` means not anonymous, i.e. it uses access keys to pull data.
    # underscore is given to this variable is for streamlit not to cache this variable in the read_file function 
    _fs = s3fs.S3FileSystem(anon=False)

    df = pq.ParquetDataset(filename, filesystem=_fs).read_pandas().to_pandas()
    df['salary.minimum'] = df['salary.minimum'].astype(float)
    df['salary.maximum'] = df['salary.maximum'].astype(float)
    if 'year_month' not in list(df.columns):
        df['year_month'] = df['year'] + '_' + df['month']
    return df

def load_lottieasset(url: str) -> json:
    """_summary_

    Args:
        url (str): _description_

    Returns:
        json: _description_
    """    
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def return_job_list(df: pd.DataFrame) -> list:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        list: _description_
    """    
    streamlit_job_list = sorted(list(df['job_title'].unique()))
    streamlit_job_list.insert(0,'All')
    return streamlit_job_list

def return_industry_list(df: pd.DataFrame) -> list:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        list: _description_
    """    
    streamlit_industry_list = sorted(list(df['industrial_class_level_0'].unique()))
    streamlit_industry_list.insert(0,'All')
    return streamlit_industry_list

def return_filter_df(df: pd.DataFrame, default_job_title: str, default_job_industry: str) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        default_job_title (str): _description_
        default_job_industry (str): _description_

    Returns:
        pd.DataFrame: _description_
    """    
    if default_job_title == 'All' and default_job_industry == 'All':
        df_graph = df.copy()
    elif default_job_title == 'All':
        df_graph = df[(df['industrial_class_level_0']==default_job_industry)].copy()
    elif default_job_industry == 'All':
        df_graph = df[(df['job_title']==default_job_title)].copy()
    else:
        df_graph = df[(df['job_title']==default_job_title) & (df['industrial_class_level_0']==default_job_industry)].copy()
    df_graph = df_graph.sort_values(['year', 'month'],
        ascending = [True, True])
    
    return df_graph

def return_df_check(df):
    row = df.shape[0]
    if row < 1:
        return False
    else:
        return True

###########################################
### FUNCTION FOR THE DIAGRAM MATPLOTLIB ###
###########################################

def update_rc():
    rc = {'figure.figsize':(8,4.5),
        'axes.facecolor':'#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'font.family': 'sans',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size' : 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12}
    plt.rcParams.update(rc)

def mat_monthly_job_posting(df, default_job_title,default_job_industry):
    df_graph = return_filter_df(df, default_job_title, default_job_industry)
    has_records = return_df_check(df_graph)
    if has_records:
        df_graph = df_graph.groupby(['year_month']).count()[['year']]
        
        update_rc()
        fig, ax = plt.subplots()
        sns.barplot(
            x=df_graph.index,
            y=df_graph.year, 
            color="#b80606", 
            ax=ax
            )
        ax.tick_params('x', labelrotation=45)
        ax.set(xlabel = 'Year/Month', ylabel = 'Number of Job Posting')
        st.pyplot(fig)

        max_job = df_graph.max()[0]
        avg_job = int(df_graph.mean()[0])
        period_max_job = df_graph[df_graph['year']==max_job].index[0]
        
        st.markdown(f"The highest job posted were {max_job} in the period of {period_max_job}.\
        The average job posted were {avg_job} per month."
        )
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def mat_industry_job_posting(df, default_job_title,default_job_industry):
    df_graph = return_filter_df(df, default_job_title, 'All')
    has_records = return_df_check(df_graph)
    if has_records:
        df_graph = df_graph.groupby(['industrial_class_level_0']).count()[['year']]
        update_rc()
        fig, ax = plt.subplots()
        sns.barplot(
            x=df_graph.index,
            y=df_graph.year,
            color="#b80606",
            ax=ax
            )
        ax.tick_params('x', labelrotation=90)
        ax.set(xlabel = 'Industry', ylabel = 'Number of Job Posting')
        st.pyplot(fig)
        df_graph['percentage'] = df_graph['year']/df_graph['year'].sum()*100
        industry_list = sorted(list(df_graph[df_graph[df_graph.sort_values('year',
            ascending=False).cumsum() < 80]['percentage'].notnull()].index))
        industry_str = '], ['.join(industry_list)

        st.markdown(f"These industry [{industry_str}], account for 80% of the job posting")
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def mat_salary_time_series(df, default_job_title, default_job_industry, salary_input):
    df_graph = return_filter_df(df, default_job_title, default_job_industry)
    has_records = return_df_check(df_graph)
    if has_records:
        if salary_input == 'Min':
            salary_type = 'salary.minimum'
        else:
            salary_type = 'salary.maximum'
        update_rc()
        fig, ax = plt.subplots()

        max_y_limit = df_graph[salary_type].mean() + df_graph[salary_type].std()*3

        df_graph = df_graph[df_graph[salary_type] <= max_y_limit]

        try:
            sns.boxplot(
                x=df_graph['year_month'],
                y=df_graph[salary_type],
                color="#b80606",
                ax=ax
                )
            ax.tick_params('x', labelrotation=45)
            ax.set_xlabel('Year/Month')
            ax.set_ylabel(f'{salary_type}')
            st.pyplot(fig)
        except:
            st.markdown(
                f"Error detected for the selected record for {default_job_title} \
                job in {default_job_industry} industry. Please select other \
                combination of job title and industry")
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def mat_salary_experience_series(df, default_job_title, default_job_industry, salary_input):
    df_graph = return_filter_df(df, default_job_title, default_job_industry)
    has_records = return_df_check(df_graph)
    if has_records:
        if salary_input == 'Min':
            salary_type = 'salary.minimum'
        else:
            salary_type = 'salary.maximum'
        update_rc()
        fig, ax = plt.subplots()

        max_y_limit = df_graph[salary_type].mean() + df_graph[salary_type].std()*3

        df_graph = df_graph[df_graph[salary_type] <= max_y_limit]

        df_graph = df_graph.sort_values(['job_exp'],
              ascending = [True])
        try:
            sns.boxplot(
                x=df_graph['job_exp'],
                y=df_graph[salary_type], 
                color='#b80606', 
                ax=ax
                )
            ax.tick_params('x', labelrotation=45)
            ax.set_xlabel('Year_of_Experiences')
            ax.set_ylabel(f'{salary_type}')
            st.pyplot(fig)
        except:
            st.markdown(
                f"Error detected for the selected record for {default_job_title} \
                job in {default_job_industry} industry. Please select other \
                combination of job title and industry")
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def mat_salary_industry_series(df, default_job_title, default_job_industry, salary_input):
    df_graph = return_filter_df(df, default_job_title, 'All')
    has_records = return_df_check(df_graph)
    if has_records:
        if salary_input == 'Min':
            salary_type = 'salary.minimum'
        else:
            salary_type = 'salary.maximum'
        update_rc()
        fig, ax = plt.subplots()

        max_y_limit = df_graph[salary_type].mean() + df_graph[salary_type].std()*3

        df_graph = df_graph[df_graph[salary_type] <= max_y_limit]

        sns.boxplot(
            x=df_graph['industrial_class_level_0'],
            y=df_graph[salary_type], 
            color='#b80606', 
            ax=ax
            )
        ax.tick_params('x', labelrotation=90)
        ax.set_xlabel('Industry')
        ax.set_ylabel(f'{salary_type}')
        st.pyplot(fig)
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

#######################################
### FUNCTION FOR THE DIAGRAM PLOTLY ###
#######################################

def plotly_monthly_job_posting(df, default_job_title,default_job_industry):
    df_graph = return_filter_df(df, default_job_title, default_job_industry)
    has_records = return_df_check(df_graph)
    if has_records:
        df_graph = df_graph.groupby(['year_month']).count()[['year']]
        df_graph.rename(columns = {'year':'job_posting_count'}, inplace = True)
        fig = px.bar(
            df_graph,
            x=df_graph.index,
            y='job_posting_count',
            color='job_posting_count'
            )

        fig.update_layout(
            xaxis_title="Year_Month",
            yaxis_title="No. of Job Posting",
            font=dict(
                family="Roboto Condensed",
                size=12,
                color="white")
        )

        st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

        max_job = df_graph.max()[0]
        avg_job = int(df_graph.mean()[0])
        period_max_job = df_graph[df_graph['job_posting_count']==max_job].index[0]
        
        st.markdown(f"The highest job posted were {max_job} in the period of {period_max_job}.\
        The average job posted were {avg_job} per month."
        )
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def plotly_industry_job_posting(df, default_job_title,default_job_industry):
    df_graph = return_filter_df(df, default_job_title, 'All')
    has_records = return_df_check(df_graph)
    if has_records:
        df_graph.rename(columns = {'industrial_class_level_0':'industry'}, inplace = True)
        df_graph = df_graph.groupby(['industry']).count()[['year']]
        df_graph.rename(columns = {'year':'job_posting_count'}, inplace = True)
        fig = px.bar(
            df_graph,
            x=df_graph.index,
            y='job_posting_count',
            color='job_posting_count'
            )

        fig.update_layout(
            xaxis_title="Industry",
            yaxis_title="No. of Job Posting",
            font=dict(
                family="Roboto Condensed",
                size=12,
                color="white"),
            height=700
        )
        fig.update_layout(hovermode='x unified')

        st.plotly_chart(fig, use_container_width=True, sharing="streamlit")

        df_graph['percentage'] = df_graph['job_posting_count']/df_graph['job_posting_count'].sum()*100
        industry_list = sorted(list(df_graph[df_graph[df_graph.sort_values('job_posting_count',
            ascending=False).cumsum() < 80]['percentage'].notnull()].index))
        industry_str = '], ['.join(industry_list)

        st.markdown(f"These industry [{industry_str}], account for 80% of the job posting")
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def plotly_salary_time_series(df, default_job_title, default_job_industry, salary_input):
    df_graph = return_filter_df(df, default_job_title, default_job_industry)
    has_records = return_df_check(df_graph)
    if has_records:
        if salary_input == 'Min':
            salary_type = 'salary.minimum'
        else:
            salary_type = 'salary.maximum'

        max_y_limit = df_graph[salary_type].mean() + df_graph[salary_type].std()*3

        df_graph = df_graph[df_graph[salary_type] <= max_y_limit]

        try:
            fig = px.box(
                df_graph,
                x='year_month',
                y=salary_type,
                notched=True
                )

            fig.update_layout(
                xaxis_title="Industry",
                yaxis_title="No. of Job Posting",
                font=dict(
                    family="Roboto Condensed",
                    size=12,
                    color="white"),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
        except:
            st.markdown(
                f"Error detected for the selected record for {default_job_title} \
                job in {default_job_industry} industry. Please select other \
                combination of job title and industry")
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def plotly_salary_experience_series(df, default_job_title, default_job_industry, salary_input):
    df_graph = return_filter_df(df, default_job_title, default_job_industry)
    has_records = return_df_check(df_graph)
    if has_records:
        if salary_input == 'Min':
            salary_type = 'salary.minimum'
        else:
            salary_type = 'salary.maximum'
        update_rc()
        fig, ax = plt.subplots()

        max_y_limit = df_graph[salary_type].mean() + df_graph[salary_type].std()*3

        df_graph = df_graph[df_graph[salary_type] <= max_y_limit]

        df_graph = df_graph.sort_values(['job_exp'],
              ascending = [True])
        try:

            fig = px.box(
                df_graph,
                x='job_exp',
                y=salary_type,
                notched=True
                )

            fig.update_layout(
                xaxis_title="Year_of_Experiences",
                yaxis_title=f'{salary_type}',
                font=dict(
                    family="Roboto Condensed",
                    size=12,
                    color="white")
            )
            st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
        except:
            st.markdown(
                f"Error detected for the selected record for {default_job_title} \
                job in {default_job_industry} industry. Please select other \
                combination of job title and industry")
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")

def plotly_salary_industry_series(df, default_job_title, default_job_industry, salary_input):
    df_graph = return_filter_df(df, default_job_title, 'All')
    has_records = return_df_check(df_graph)
    if has_records:
        if salary_input == 'Min':
            salary_type = 'salary.minimum'
        else:
            salary_type = 'salary.maximum'
        update_rc()
        fig, ax = plt.subplots()

        max_y_limit = df_graph[salary_type].mean() + df_graph[salary_type].std()*3

        df_graph = df_graph[df_graph[salary_type] <= max_y_limit]

        fig = px.box(
            df_graph,
            x='industrial_class_level_0',
            y=salary_type,
            notched=True
            )

        fig.update_layout(
            xaxis_title="industrial_class_level_0",
            yaxis_title=f'{salary_type}',
            font=dict(
                family="Roboto Condensed",
                size=12,
                color="white"),
            height=700
        )
        st.plotly_chart(fig, use_container_width=True, sharing="streamlit")
    else:
        st.markdown(
            f"There is not record for {default_job_title} job in {default_job_industry} industry")


##############################
### SETTING UP THE WEB APP ###
##############################

# Configures the default settings of the page to wide which make use of the whole screen
st.set_page_config(layout="wide")
# Download the animation from lottiefiles for use in the web app
lottie_book = load_lottieasset('https://assets10.lottiefiles.com/packages/lf20_ncbuzxm7.json')
# Load the animation into the web app
st_lottie(lottie_book, speed=1, height=200, key="initial")

s3_bucket = 'mcf'
s3_key = '/streamlit/mcf_streamlit.parquet'

df = setup_data("s3://mcf/streamlit/mcf_streamlit.parquet")

# Setting the matplotlib backend to 'agg'
matplotlib.use("agg")
# 
_lock = RendererAgg.lock
# Setting seaborn style as darkgrid
sns.set_style('darkgrid')

#####################################################
### SETTING THE CUSTOM FONT FOR STREAMLIT WEB APP ###
#####################################################

streamlit_style = """
			<style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Roboto Condensed', sans-serif;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

#####################################
### SETTING THE STREAMLIT SIDEBAR ###
#####################################

### Job Title Selection
st.sidebar.markdown("**To begin, please select one of the job title and industry** 👇")
streamlit_job_list = return_job_list(df)
default_job_title = st.sidebar.selectbox("Select one of the job title", (streamlit_job_list), key='default_job_title')

### Industry Selection
streamlit_industry_list = return_industry_list(df)
default_job_industry = st.sidebar.selectbox("Select one of the industry", (streamlit_industry_list), key='default_job_industry')

### 
salary_input = st.sidebar.select_slider("Select Min / Max Salary:", ["Min", "Max"])

temp_df = return_filter_df(df, default_job_title, default_job_industry)

###
st.sidebar.markdown("**You can download the data of the selected job title and industry by clicking the following button** 👇")
st.sidebar.markdown(download_df(temp_df), unsafe_allow_html=True)

#####################################
### Setting the streamlit Columns ###
#####################################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 1, .1, 1, .1))
#
row0_1.title('Analyzing IT Job Posting')
#
with row0_2:
    st.write('')
#
row0_2.subheader(
    'A Streamlit web app by [Jeffrey], visit my github at (https://github.com/jeffrey-siew)')

############### Setting the streamlit Columns
line1_spacer1, line1_1, line1_spacer2 = st.columns((.1, 1, .1))

with line1_1:
    st.header(f'Analyzing the Job Posting History of: **{default_job_title}** job in **{default_job_industry}** industry')

line2_spacer1, line2_1, line2_spacer2 = st.columns((.1, 1, .1))
with line2_1:
    st.markdown("Hey there! Welcome to Jeffrey's Job Post Analysis App. If you're on a mobile device, remain in portrait view are preferred. Give it a go!")
    st.markdown("The sidebar house the filter options and ability to download the dataframe")
    st.markdown("This web app is to showcase the capability of streamlit to delivery data products to end user.")
    st.markdown("Streamlit are extremely versatile and fully pythonic")

############### Setting the streamlit Columns
row31_space1, row31_1, row31_space2, row31_2, row31_space3 = st.columns(
    (.2, 2.3, .4, 4.4, .2))
with row31_1:
    st.markdown('Streamlit supports a wide range of visualization tools, go ahead to try any one of them for this web app, 👉')
with row31_2:
    plot_list = ['MatPlotLib', 'Plotly']
    default_plot = st.selectbox("Select one of plotting framework", (plot_list), key='default_plot')

############### Setting the streamlit Columns
row2_spacer1, row2_1, row2_spacer2 = st.columns((.2, 7.1, .2))
with row2_1:
    st.subheader('Analysis Job Posting via Monthly Period')

row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (.2, 2.3, .4, 4.4, .2))
with row3_1, _lock:
    st.subheader("Monthly Job Posting Count")
    st.markdown('Investigate the monthly job posting for selected job and industry')
with row3_2, _lock:
    if default_plot == 'MatPlotLib':
        mat_monthly_job_posting(df, default_job_title,default_job_industry)
    elif default_plot == 'Plotly':
        plotly_monthly_job_posting(df, default_job_title,default_job_industry)

############### Setting the streamlit Columns
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('Analysis Job Posting via Industry')

row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns(
    (.2, 2.3, .4, 4.4, .2))

with row5_1, _lock:
    st.subheader("Industry Job Posting Count")
    st.markdown('Investigate the industry job posting for selected job')
with row5_2, _lock:
    if default_plot == 'MatPlotLib':
        mat_industry_job_posting(df, default_job_title,default_job_industry)
    elif default_plot == 'Plotly':
        plotly_industry_job_posting(df, default_job_title,default_job_industry)


############### Setting the streamlit Columns
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Analysis via Time Series')

row7_space1, row7_1, row7_space2, row7_2, row7_space3 = st.columns(
    (.2, 2.3, .4, 4.4, .2))

with row7_1, _lock:
    if salary_input == 'Min':
        st.subheader("Min Salary Time Series")
    else:
        st.subheader("Max Salary Time Series")
    st.markdown(f'Investigate the ({salary_input}) time series for selected job')
with row7_2, _lock:
    if default_plot == 'MatPlotLib':
        mat_salary_time_series(df, default_job_title, default_job_industry, salary_input)
    elif default_plot == 'Plotly':
        plotly_salary_time_series(df, default_job_title, default_job_industry, salary_input)


############### Setting the streamlit Columns
row8_spacer1, row8_1, row8_spacer2 = st.columns((.2, 7.1, .2))
with row8_1:
    st.subheader('Analysis via Year of Experiences')

row9_space1, row9_1, row9_space2, row9_2, row9_space3 = st.columns(
    (.2, 2.3, .4, 4.4, .2))

with row9_1, _lock:
    if salary_input == 'Min':
        st.subheader("Min Salary Experiences Comparison")
    else:
        st.subheader("Max Salary Experiences Comparison")
    st.markdown(f'Investigate the ({salary_input}) time series for selected job')
with row9_2, _lock:
    if default_plot == 'MatPlotLib':
        mat_salary_experience_series(df, default_job_title, default_job_industry, salary_input)
    elif default_plot == 'Plotly':
        plotly_salary_experience_series(df, default_job_title, default_job_industry, salary_input)


############### Setting the streamlit Columns
row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
with row10_1:
    st.subheader('Analysis via Industry Comparison')

row11_space1, row11_1, row11_space2, row11_2, row11_space3 = st.columns(
    (.2, 2.3, .4, 4.4, .2))

with row11_1, _lock:
    if salary_input == 'Min':
        st.subheader("Min Salary Industry Comparison")
    else:
        st.subheader("Max Salary Industry Comparison")
    st.markdown(f'Investigate the ({salary_input}) time series for selected job')
with row11_2, _lock:
    if default_plot == 'MatPlotLib':
        mat_salary_industry_series(df, default_job_title, default_job_industry, salary_input)
    elif default_plot == 'Plotly':
        plotly_salary_industry_series(df, default_job_title, default_job_industry, salary_input)
    


