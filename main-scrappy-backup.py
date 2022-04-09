import streamlit as st
import pandas as pd
import numpy as np
import io
# from io import BytesIO
# import xlsxwriter
from deep_translator import GoogleTranslator
from titlecase import titlecase
import warnings
import altair as alt
import requests
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import requests
from requests.structures import CaseInsensitiveDict

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="MIG Data Cleaning App", page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png")
hide_menu_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# def moreover_url(oldURLs):
#     # REWRITE TO ONLY HIT ONLINE/BLOG TYPE
#     # st.write(f"Moreovers before: {len(data.loc[data['URL'].str.contains("moreover", na=False)]})
#     user_agent = {'User-agent': '14.0.3 Safari'}
#     session = requests.Session()
#     for x in oldURLs:
#       substring = 'ct.moreover'
#       if substring in x:
#         try:
#             newURL = session.get(x, header=user_agent)
#             newURLs.append(newURL.url)
#             # print(newURL.url)
#         except:
#             # print(x)
#             newURLs.append(x)
#       else:
#         # print(x)
#         newURLs.append(x)
#     data['Updated URL'] = newURLs
    # st.write(f"Moreovers after: {len(data.loc[data['Updated URL'].str.contains("moreover", na=False)]})


def yahoo_cleanup(url_string):
  data.loc[data['URL'].str.contains(url_string, na=False), "Outlet"] = "Yahoo! News"
  data.loc[data['URL'].str.contains(url_string, na=False), "Impressions"] = 80828000
  data.loc[data['URL'].str.contains(url_string, na=False), "Country"] = (np.nan)
  data.loc[data['URL'].str.contains(url_string, na=False), "Continent"] = (np.nan)
  data.loc[data['URL'].str.contains(url_string, na=False), "City"] = (np.nan)
  data.loc[data['URL'].str.contains(url_string, na=False), "Prov/State"] = (np.nan)
  data.loc[data['URL'].str.contains(url_string, na=False), "sub"] = data['URL'].str.rsplit('/', 1).str[-1]
  data.loc[data['URL'].str.contains(url_string, na=False), "URL"] = 'https://news.yahoo.com/'+data["sub"]
  data.drop(["sub"], axis = 1, inplace=True, errors='ignore')


def top_x_by_mentions(df, column_name):
  """Returns top N items by mention count"""
  x_table = pd.pivot_table(df, index=column_name, values=["Mentions"], aggfunc="count")
  x_table = x_table.sort_values("Mentions", ascending=False)
  x_table = x_table.rename(columns={"Mentions": "Hits"})
  return x_table.head(10)


def fixable_impressions_list(df):
    """WIP - Returns item from most fixable imp list"""
    imp_table = pd.pivot_table(df, index="Outlet", values=["Mentions", "Impressions"], aggfunc="count")
    imp_table["Missing"] = imp_table["Mentions"] - imp_table["Impressions"]
    imp_table = imp_table[imp_table["Impressions"] > 0]
    imp_table = imp_table[imp_table['Missing'] > 0]
    imp_table = imp_table.sort_values("Missing", ascending=False)
    imp_table = imp_table.reset_index()
    return imp_table


def fix_imp(df, outlet, new_impressions_value):
    """Updates all impressions for a given outlet"""
    df.loc[df["Outlet"] == outlet, "Impressions"] = new_impressions_value


def outlet_imp(df, outlet):
    """Returns the various authors for a given headline"""
    outlet_imps = (df[df.Outlet == outlet].Impressions.value_counts().reset_index())
    return outlet_imps


def fix_author(df, headline_text, new_author):
    """Updates all authors for a given headline"""
    df.loc[df["Headline"] == headline_text, "Author"] = new_author


def fixable_headline_stats(df, primary="Headline", secondary="Author"):
    """tells you how many author fields can be fixed and other stats"""
    total = df["Mentions"].count()
    headline_table = pd.pivot_table(df, index=primary, values=["Mentions", secondary], aggfunc="count")
    headline_table["Missing"] = headline_table["Mentions"] - headline_table[secondary]
    missing = headline_table.Missing.sum()
    headline_table = headline_table[headline_table[secondary] > 0]
    headline_table = headline_table[headline_table['Missing'] > 0]
    fixable = headline_table.Missing.sum()
    fixable_headline_count = headline_table.Missing.count()
    total_known = total - missing
    percent_known = "{:.0%}".format((total_known) / total)
    percent_knowable = "{:.0%}".format((total - (missing - fixable)) / total)
    stats = (
        f"Total rows: \t\t{total} \nTotal Known: \t\t{total_known}\nPercent Known: \t\t{percent_known} \nFixable Fields: \t{fixable}\nUnique Fixable: \t{fixable_headline_count}\nPercent knowable: \t{percent_knowable}")
    return stats


def fixable_author_headline_list():
    """WIP - Returns item from most fixable headline list"""
    headline_table = pd.pivot_table(traditional, index="Headline", values=["Mentions", "Author"], aggfunc="count")
    headline_table["Missing"] = headline_table["Mentions"] - headline_table["Author"]
    headline_table = headline_table[headline_table["Author"] > 0]
    headline_table = headline_table[headline_table['Missing'] > 0]
    headline_table = headline_table.sort_values("Missing", ascending=False)
    headline_table = headline_table.reset_index()
    return headline_table


def headline_authors(df, headline_text):
    """Returns the various authors for a given headline"""
    headline_authors = (df[df.Headline == headline_text].Author.value_counts().reset_index())
    return headline_authors


def author_matcher(counter):
    temp_headline_list = fixable_author_headline_list()
    headline_text = temp_headline_list.iloc[counter]['Headline']
    st.subheader("Most Fixable Headline")
    st.write(headline_text)
    st.subheader("Possible Authors")
    st.write(headline_authors(traditional, headline_text))
    with st.form('auth updater'):
        new_author = st.text_input("\nWhat name should be applied to the author field? \n")
        submitted = st.form_submit_button("Update Author")
        if submitted:
            fix_author(traditional, headline_text, new_author)
        st.session_state.df_traditional = traditional

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Top Authors")
        st.write(original_top_authors)

    with col2:
        st.subheader("New Top Authors")
        st.write(top_x_by_mentions(traditional, "Author"))
    st.experimental_rerun()


def translate_col(df, name_of_column):
    """Replaces non-English string in column with English"""
    global dictionary
    dictionary = {}
    unique_non_eng = list(set(df[name_of_column][df['Language'] != 'English'].dropna()))
    if '' in unique_non_eng:
        unique_non_eng.remove('')
    with st.spinner('Running translation now...'):
        with ThreadPoolExecutor(max_workers=30) as ex:
            results = ex.map(translate, [text for text in unique_non_eng])
    df[name_of_column].replace(dictionary, inplace=True)


def translate(text):
    dictionary[text] = (GoogleTranslator(source='auto', target='en').translate(text[:1500]))


def translation_stats_combo():
    non_english_records = len(traditional[traditional['Language'] != 'English']) + len(social[social['Language'] != 'English'])
    minutes = non_english_records//100
    if minutes == 0:
        min_word = 'minute'
    else:
        min_word = 'minutes'
    st.write(f"There are {non_english_records} non-English records in your data.")
    st.write(f"\nAllow around {minutes}-{minutes + 1} {min_word} per column for translation.")


def fetch_outlet(author_name):
  contact_url = "https://mediadatabase.agilitypr.com/api/v4/contacts/search"
  headers = CaseInsensitiveDict()
  headers["Content-Type"] = "text/json"
  headers["Accept"] = "text/json"
  headers["Authorization"] = st.secrets["authorization"]
  headers["client_id"] = st.secrets["client_id"]
  headers["userclient_id"] = st.secrets["userclient_id"]

  data_a = '''
  {  
    "aliases": [  
      "'''

  data_b = '''"  
    ]   
  }
  '''

  data = data_a + author_name + data_b
  contact_resp = requests.post(contact_url, headers=headers, data=data)

  return contact_resp.json()

  # if contact_resp.json()['results'] == []:
  #   print("No match found")
  # else:
  #   print(contact_resp.json()['results'][0]['firstName'], contact_resp.json()['results'][0]['lastName'])
  #   print(contact_resp.json()['results'][0]['primaryEmployment']['jobTitle'])
  #   print(contact_resp.json()['results'][0]['primaryEmployment']['outletName'])
  #   print(contact_resp.json()['results'][0]['country']['name'])

format_dict = {'AVE':'${0:,.0f}', 'Audience Reach': '{:,d}', 'Impressions': '{:,d}'}

if 'page' not in st.session_state:
    st.session_state['page'] = '1: Upload your CSV'
if 'page_subtitle' not in st.session_state:
    st.session_state.page_subtitle = ''
if 'export_name' not in st.session_state:
    st.session_state.export_name = ''
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = pd.DataFrame()
if 'df_uncleaned' not in st.session_state:
    st.session_state.df_uncleaned = pd.DataFrame()
if 'df_traditional' not in st.session_state:
    st.session_state.df_traditional = pd.DataFrame()
if 'df_social' not in st.session_state:
    st.session_state.df_social = pd.DataFrame()
if 'df_dupes' not in st.session_state:
    st.session_state.df_dupes = pd.DataFrame()
if 'upload_step' not in st.session_state:
    st.session_state.upload_step = False
if 'standard_step' not in st.session_state:
    st.session_state.standard_step = False
if 'outliers' not in st.session_state:
    st.session_state.outliers = False
if 'original_auths' not in st.session_state:
    st.session_state.original_auths = pd.DataFrame()
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'translated_headline' not in st.session_state:
    st.session_state.translated_headline = False
if 'translated_summary' not in st.session_state:
    st.session_state.translated_summary = False
if 'translated_snippet' not in st.session_state:
    st.session_state.translated_snippet = False
if 'filled' not in st.session_state:
    st.session_state.filled = False
if 'original_trad_auths' not in st.session_state:
    st.session_state.original_trad_auths = pd.DataFrame()
if 'author_outlets' not in st.session_state:
    st.session_state.author_outlets = None
if'auth_counter' not in st.session_state:
    st.session_state.auth_counter = 0
if'auth_outlet_table' not in st.session_state:
    st.session_state.auth_outlet_table = pd.DataFrame()


# Sidebar and page selector
st.sidebar.image('https://agilitypr.news/images/Agility-centered.svg', width=200)
st.sidebar.title('MIG: Data Cleaning App')
pagelist = [
    "1: Getting Started",
    "2: Standard Cleaning",
    "3: Impressions - Outliers",
    "4: Impressions - Fill Blanks",
    "5: Authors",
    "5.5: Author - Outlets",
    "6: Translation",
    "7: Review",
    "8: Download"]

page = st.sidebar.radio("Data Cleaning Steps:", pagelist, index=0)
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.caption("v.1.5.1.3")

if page == "1: Getting Started":
    st.title('Getting Started')

    if st.session_state.upload_step == True:
        st.success('File uploaded.')
        if st.button('Start Over?'):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
        data = st.session_state.df_raw
        st.header('Exploratory Data Analysis')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Metrics")
            st.metric(label="Mentions", value="{:,}".format(len(data.dropna(thresh=3))))
            st.metric(label="Impressions", value="{:,}".format(data['Audience Reach'].sum()))
        with col2:
            st.subheader("Media Type")
            st.write(data['Media Type'].value_counts())

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Top Authors")
            original_top_authors = (top_x_by_mentions(data, "Author"))
            st.write(original_top_authors)
            st.session_state.original_auths = original_top_authors
        with col4:
            st.subheader("Top Outlets")
            original_top_outlets = (top_x_by_mentions(data, "Outlet"))
            st.write(original_top_outlets)
        #
        # source = data['Sentiment'].value_counts().reset_index()
        # sentiment = alt.Chart(source).mark_arc().encode(
        #     theta=alt.Theta(field="Sentiment", type="quantitative"),
        #     color=alt.Color(field="index", type="nominal")
        # )
        #
        # st.altair_chart(sentiment, use_container_width=True)

        st.markdown('##')
        st.subheader('Mention Trend')

        trend = alt.Chart(data).mark_line().encode(
            x='Published Date:T',
            y='count(Mentions):Q'
        )
        st.altair_chart(trend, use_container_width=True)

        st.markdown('##')
        st.subheader('Impressions Trend')
        trend2 = alt.Chart(data).mark_line().encode(
            x='Published Date:T',
            y='sum(Audience Reach):Q'
        )
        st.altair_chart(trend2, use_container_width=True)

        st.subheader("Raw Data")
        st.dataframe(data.style.format(format_dict))
        st.markdown('##')

        with st.expander('Data set stats'):
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    else:
        with st.form("my_form"):
            client = st.text_input('Client organization name*', placeholder='eg. Air Canada', key='client', help='Required to build export file name.')
            period = st.text_input('Reporting period or focus*', placeholder='eg. March 2022', key='period', help='Required to build export file name.')
            uploaded_file = st.file_uploader(label='Upload your CSV*', type='csv',
                                             accept_multiple_files=False, help='Only use CSV files exported from the Agility Platform.')

            submitted = st.form_submit_button("Submit")
            if submitted and (client == "" or period == "" or uploaded_file == None):
                 st.error('Missing required form inputs above.')

            elif submitted:
                with st.spinner("Converting file format."):
                    data = pd.read_csv(uploaded_file)
                    data = data.dropna(thresh=2)

                    st.session_state.df_uncleaned = data
                    data["Mentions"] = 1

                    st.session_state.df_raw = data
                    st.session_state.upload_step = True

                    data['Audience Reach'] = data['Audience Reach'].astype('Int64')
                    data['AVE'] = data['AVE'].fillna(0)
                    st.session_state.export_name = f"{client} - {period} - clean_data.xlsx"
                    st.session_state.df_raw = data
                    st.experimental_rerun()


elif page == "2: Standard Cleaning":
    st.title('Standard Cleaning')
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step:
        st.success("Standard cleaning already done!")
        traditional = st.session_state.df_traditional
        social = st.session_state.df_social
        dupes = st.session_state.df_dupes
        if len(traditional) > 0:
            with st.expander("Traditional"):
                st.dataframe(traditional.style.format(format_dict))
        if len(social) > 0:
            with st.expander("Social"):
                st.dataframe(social.style.format(format_dict))
        if len(dupes) > 0:
            with st.expander("Deleted Duplicates"):
                st.dataframe(dupes.style.format(format_dict))
    else:
        data = st.session_state.df_raw
        # data['Published Date'] = pd.to_datetime(data['Published Date'])
        # data = data.rename(columns={
        #     'Published Date': 'Date'})
        # min_date = (data.Date.min().date())
        # max_date = (data.Date.max().date())
        # # st.write(min_date)
        # # st.write(max_date)
        # st.write(min_date).floor('D')
        #
        # date_range = st.date_input('Date Range', value=[min_date, max_date], min_value=min_date, max_value=max_date)
        # #
        # st.write(type(min_date))
        # st.write(type(data.Date[4]))

        with st.form("my_form_basic_cleaning"):
            st.subheader("Cleaning options")
            merge_online = st.checkbox("Merge 'blogs' and 'press releases' into 'Online'", value=True)
            fill_known_imp = st.checkbox("Fill missing impressions values where known match exists in data", value=True)
            # remove_newswires = st.checkbox("Remove newswires from data")
            # collect_moreovers = st.checkbox("Collect original urls from moreover links.", help="Can take a few minutes")
            # st.subheader("Adjust the Date Range")
            # # date_range = st.date_input('Date Range', value=[min_date, max_date], min_value=min_date, max_value=max_date)
            # start_date = st.date_input('Start Date', value=min_date, min_value=min_date, max_value=max_date)
            # end_date = st.date_input('End Date', value=max_date, min_value=min_date, max_value=max_date)
            submitted = st.form_submit_button("Go!")
            if submitted:
            #     # greater than the start date and smaller than the end date
            #     mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
            #     data = data.loc[mask]
                with st.spinner("Running standard cleaning."):

                    data = data.rename(columns={
                        'Published Date': 'Date',
                        'Published Time': 'Time',
                        'Media Type': 'Type',
                        'Coverage Snippet': 'Snippet',
                        'Province/State': 'Prov/State',
                        'Audience Reach': 'Impressions'})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write('✓ Columns Renamed')

                    data.Type.replace({
                        "ONLINE_NEWS": "ONLINE NEWS",
                        "PRESS_RELEASE": "PRESS RELEASE"}, inplace=True)

                    if merge_online:
                        data.Type.replace({
                            "ONLINE NEWS": "ONLINE",
                            "PRESS RELEASE": "ONLINE",
                            "BLOGS": "ONLINE"}, inplace=True)
                    data.loc[data['URL'].str.contains("www.facebook.com", na=False), 'Type'] = "FACEBOOK"
                    data.loc[data['URL'].str.contains("/twitter.com", na=False), 'Type'] = "TWITTER"
                    data.loc[data['URL'].str.contains("www.instagram.com", na=False), 'Type'] = "INSTAGRAM"
                    data.loc[data['URL'].str.contains("reddit.com", na=False), 'Type'] = "REDDIT"
                    data.loc[data['URL'].str.contains("youtube.com", na=False), 'Type'] = "YOUTUBE"
                    with col2:
                        st.write('✓ Media Types Cleaned')

                    if "Original URL" in data:
                        data.loc[data["Original URL"].notnull(), "URL"] = data["Original URL"]
                        with col3:
                            st.write('✓ Original URLs Merged')


                    data.drop(["Timezone",
                               "Word Count",
                               "Duration",
                               "Original URL",
                               "Image URLs",
                               "Folders",
                               "Notes",
                               "County",
                               "isAudienceFromPartnerUniqueVisitor"],
                              axis=1, inplace=True, errors='ignore')
                    with col1:
                        st.write('✓ Junk Columns Dropped')

                    # Move columns
                    temp = data.pop('Impressions')
                    data.insert(5, 'Impressions', temp)
                    # data.Impressions = data.Impressions.astype('Int64')
                    temp = data.pop('Mentions')
                    data.insert(5, 'Mentions', temp)
                    with col2:
                        st.write('✓ Columns Sorted')

                    # Strip extra white space
                    data['Headline'] = data['Headline'].astype(str)
                    data['Headline'].str.strip()
                    data['Outlet'].str.strip()
                    data['Author'].str.strip()
                    data['Headline'] = data['Headline'].str.replace('   ', ' ')
                    data['Outlet'] = data['Outlet'].str.replace('   ', ' ')
                    data['Author'] = data['Author'].str.replace('   ', ' ')
                    data['Headline'] = data['Headline'].str.replace('  ', ' ')
                    data['Outlet'] = data['Outlet'].str.replace('  ', ' ')
                    data['Author'] = data['Author'].str.replace('  ', ' ')
                    with col3:
                        st.write('✓ Extra Spaces Removed')

                    # Remove (Online)
                    data['Outlet'] = data['Outlet'].str.replace(' \(Online\)', '')
                    with col1:
                        st.write('✓ "(Online)" Removed')

                    # Tag exploder
                    if "Tags" in data:
                        data['Tags'] = data['Tags'].astype(str) # needed if column there but all blank
                        data = data.join(data["Tags"].str.get_dummies(sep=","))
                        with col2:
                            st.write('✓ Tags Expanded')

                    # SOCIALS To sep df
                    soc_array = ['FACEBOOK', 'TWITTER', 'INSTAGRAM', 'REDDIT', 'YOUTUBE']
                    social = data.loc[data['Type'].isin(soc_array)]
                    data = data[~data['Type'].isin(soc_array)]

                    with col3:
                        st.write('✓ Social Split Out')


                    # Fill known impressions
                    if fill_known_imp:
                        imp_fix_table = fixable_impressions_list(data)
                        # st.table(imp_fix_table)
                        # st.write(f"known impressions to fill: {len(imp_fix_table)}")
                        # known_filled = 0

                        for outlet in imp_fix_table.Outlet:
                            # print(outlet, len(outlet_imp(data, outlet)), int(outlet_imp(data, outlet)['index']))
                            if len(outlet_imp(data, outlet)) == 1:
                                fix_imp(data, outlet, int(outlet_imp(data, outlet)['index']))
                                # known_filled += 1

                            with col1:
                                st.write(f"✓ Filled Known Impressions")

                    # TODO: remove newswires
                    # if remove_newsires:
                    #     # save the newswires df & excel sheet
                    #     # based on text search of summary for known newswire names.
                    #     pass


                    # AP Cap
                    broadcast_array = ['RADIO', 'TV']
                    broadcast = data.loc[data['Type'].isin(broadcast_array)]
                    data = data[~data['Type'].isin(broadcast_array)]

                    data[['Headline']] = data[['Headline']].fillna('')
                    data['Headline'] = data['Headline'].map(lambda Headline: titlecase(Headline))

                    with col1:
                        st.write('✓ AP Style Capitalization')


                    ### COLUMN BASED DUPLICATE REMOVAL ####

               #      """
               #      GOALS
               #      - Keep most data rich version
               #      - Save everything deleted to dupes dataframe
               #      - Drop duplicates based on the following conditions:
               #          - excluding broadcast and social
               #          - none of TYPE + OUTLET + HEADLINE are blank
               #          - TYPE + OUTLET + HEADLINE are duplicated across records
               #
               #      PROCESS ASSUMPTIONS
               #      - broadcast and social already split out
               #      - AP capitalization applied to headlines of the rest
               #
               #      PROCESS
               #      - pull aside all records with blank headlines, types, and outlets
               #      - create dupe_helper column with type/outlet/headline
               #      - sort to prioritize records with author, impression, AVE to first in dupe sets
               #      - dupe_cols = data[data['dupe_helper'].duplicated(keep='first') == True]
               #      - data drop duplicates based on same
               #      - join blank headlines/types/outlets back in.
               #
               #      dupe_cols.sort_values(["dupe_helper", "Author", "Impressions", "AVE"],
               # axis = 0, ascending = [True, True, False, False])
               #
               #      """

                    blank_set = data[data.Headline.isna() | data.Outlet.isna() | data.Type.isna()]
                    data = data[~data.Headline.isna() | ~data.Outlet.isna() | ~data.Type.isna()]

                    data["dupe_helper"] = data['Type'] + data['Outlet'] + data['Headline']  # make the helper column
                    data = data.sort_values(["dupe_helper", "Author", "Impressions", "AVE"], axis=0,
                                            ascending=[True, True, False, False])
                    dupe_cols = data[data['dupe_helper'].duplicated(keep='first') == True]
                    data = data[~data['dupe_helper'].duplicated(keep='first') == True]

                    # Drop dupe helper column from both
                    data.drop(["dupe_helper"], axis=1, inplace=True, errors='ignore')
                    dupe_cols.drop(["dupe_helper"], axis=1, inplace=True, errors='ignore')

                    frames = [data, blank_set]
                    data = pd.concat(frames)


                    ### END SECTION ###


                    # TODO: Flag suspected duplicates based on Outlet, Type, and fuzzy match headline


                    # Yahoo standardizer
                    yahoo_cleanup('sports.yahoo.com')
                    yahoo_cleanup('www.yahoo.com')
                    yahoo_cleanup('news.yahoo.com')
                    yahoo_cleanup('style.yahoo.com')
                    yahoo_cleanup('finance.yahoo.com')
                    with col2:
                        st.write('✓ Yahoo Standardization')

                    # Set aside blank URLs
                    blank_urls = data[data.URL.isna()]
                    data = data[~data.URL.isna()]


                    # Add temporary dupe URL helper column
                    data['URL_Helper'] = data['URL'].str.lower()
                    data['URL_Helper'] = data['URL_Helper'].str.replace('http:', 'https:')

                    # Sort and Save duplicate URLS
                    data = data.sort_values(["URL_Helper", "Author", "Impressions", "AVE"], axis=0,
                                            ascending=[True, True, False, False])
                    dupe_urls = data[data['URL_Helper'].duplicated(keep='first') == True]

                    # Drop duplicate URLs
                    data = data[~data['URL_Helper'].duplicated(keep='first') == True]

                    # Drop Helper columns from both
                    data.drop(["URL_Helper"], axis=1, inplace=True, errors='ignore')
                    dupe_urls.drop(["URL_Helper"], axis=1, inplace=True, errors='ignore')


                    # Drop helper column and rejoin broadcast
                    frames = [data, broadcast, blank_set, blank_urls]
                    traditional = pd.concat(frames)
                    dupes = pd.concat([dupe_urls, dupe_cols])


                    with col3:
                        st.write('✓ Duplicates Removed')

                    if len(traditional) > 0:
                        with st.expander("Traditional"):
                            st.dataframe(traditional.style.format(format_dict))
                    if len(social) > 0:
                        with st.expander("Social"):
                            st.dataframe(social.style.format(format_dict))
                    if len(dupes) > 0:
                        with st.expander("Deleted Duplicates"):
                            st.dataframe(dupes.style.format(format_dict))

                    original_trad_auths = top_x_by_mentions(traditional, "Author")
                    st.session_state.original_trad_auths = original_trad_auths
                    st.session_state.df_traditional = traditional
                    st.session_state.df_social = social
                    st.session_state.df_dupes = dupes
                    st.session_state.standard_step = True

                    # Fetch remaining Moreover URLs
                    # if collect_moreovers:
                    #     # Put moreovers in sep. DF:
                    #     moreovers = data.loc[data['URL'].str.contains('ct.moreover', na=False)]
                    #
                    #     # Remove moreovers from main DF
                    #     data = data[~data['URL'].str.contains('ct.moreover', na=False)]
                    #
                    #     # Make lists for oldurls and new
                    #     oldURLs = moreovers['URL']
                    #     newURLs = []
                    #     USER_AGENTS = (
                    #         'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0',
                    #         'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100 101 Firefox/22.0',
                    #         'Mozilla/5.0 (Windows NT 6.1; rv:11.0) Gecko/20100101 Firefox/11.0',
                    #       )
                    #
                    #     # loop through them
                    #     for address in oldURLs:
                    #         substring = 'ct.moreover'
                    #         if substring in address:
                    #             try:
                    #                 # user_agent = {'User-agent': '14.0.3 Safari'}
                    #                 session = requests.Session()
                    #                 r1 = session.get(address, timeout=3, headers={'User-Agent': random.choice(USER_AGENTS)})
                    #                 # st.write(r1.url)
                    #                 newURLs.append(r1.url)
                    #             except:
                    #                 newURLs.append(address)
                    #
                    #     moreovers['Updated URL'] = newURLs
                    #     moreovers.loc[moreovers["URL"].notnull(), "URL"] = moreovers["Updated URL"]
                    #
                    #     frames = [data, moreovers]
                    #     data = pd.concat(frames)
                    #
                    #     with col3:
                    #         st.write('✓ Moreover URLs searched')

elif page == "3: Impressions - Outliers":
    traditional = st.session_state.df_traditional
    st.title('Impressions - Outliers')
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif len(traditional) == 0:
        st.subheader("No traditional media in data. Skip to next step.")
    elif st.session_state.standard_step == False:
            st.error('Please run the Standard Cleaning before trying this step.')
    else:
        st.subheader('Check highest impressions numbers:')
        outliers = traditional[['Outlet', 'Type', 'Impressions', 'Headline', 'URL', 'Country']].nlargest(100, 'Impressions')
        outliers.index.name = 'Row'
        st.dataframe(outliers.style.format(format_dict))
        outlier_index = outliers.index.values.tolist()

        with st.form("Update Outliers", clear_on_submit=True):
            st.subheader("Update Impressions Outliers")
            index_numbers = st.multiselect('Row index number(s): ', outlier_index,
                                               help='Select the row number from the table above.')
            new_impressions_value = int(st.number_input('New impressions value for row(s)', step=1, help='Write in the new impression value for the selected row.'))
            submitted = st.form_submit_button("Go!")
            if submitted:
                for index_number in index_numbers:
                    traditional.loc[int(index_number), "Impressions"] = new_impressions_value
                st.session_state.df_traditional = traditional
                st.session_state.outliers = True
                st.experimental_rerun()


elif page == "4: Impressions - Fill Blanks":
    st.title('Impressions - Fill Blanks')
    traditional = st.session_state.df_traditional

    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')

    elif len(traditional) == 0:
        st.subheader("No traditional media in data. Skip to next step.")

    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')

    elif st.session_state.filled == True:
        st.success("Missing impressions fill complete!")

    elif st.session_state.outliers == False:
        st.warning('Please confirm outliers step is complete before running this step.')
        done_outliers = st.button('Done with outliers')
        if done_outliers:
            st.session_state.outliers = True
            st.experimental_rerun()

    else:
        traditional = st.session_state.df_traditional

        blank_impressions = traditional['Impressions'].isna().sum()
        mean = "{:,}".format(int(traditional.Impressions.mean()))
        median = "{:,}".format(int(traditional.Impressions.median()))
        tercile = "{:,}".format(int(traditional.Impressions.quantile(0.33)))
        quartile = "{:,}".format(int(traditional.Impressions.quantile(0.25)))
        twentieth_percentile = "{:,}".format(int(traditional.Impressions.quantile(0.2)))
        eighteenth_percentile = "{:,}".format(int(traditional.Impressions.quantile(0.18)))
        seventeenth_percentile = "{:,}".format(int(traditional.Impressions.quantile(0.17)))
        fifteenth_percentile = "{:,}".format(int(traditional.Impressions.quantile(0.15)))
        decile = "{:,}".format(int(traditional.Impressions.quantile(0.1)))

        st.markdown(f"#### MISSING: {blank_impressions}")
        st.write("*************")

        col1, col2 = st.columns(2)
        with col1:

            st.write(f"Average: {mean}")
            st.write(f"Median: {median}")
            st.write(f"Tercile: {tercile}")
            st.write(f"Quartile: {quartile}")
            st.write(f"20th Percentile: {twentieth_percentile}")
            st.write(f"18th Percentile: {eighteenth_percentile}")
            st.write(f"17th Percentile: {seventeenth_percentile}")
            st.write(f"15th Percentile: {fifteenth_percentile}")
            st.write(f"Decile: {decile}")

        with col2:
            filldict = {
                'Tercile': int(traditional.Impressions.quantile(0.33)),
                'Quartile': int(traditional.Impressions.quantile(0.25)),
                '20th percentile': int(traditional.Impressions.quantile(0.2)),
                '18th percentile': int(traditional.Impressions.quantile(0.18)),
                '17th percentile': int(traditional.Impressions.quantile(0.17)),
                '15th percentile': int(traditional.Impressions.quantile(0.15)),
                'Decile': int(traditional.Impressions.quantile(0.1))
            }
            with st.form('Fill Blanks'):
                st.subheader("Fill Blank Impressions")
                fill_blank_impressions_with = st.radio('Pick your statistical fill value: ', filldict.keys(), index=5)
                submitted = st.form_submit_button("Fill Blanks")
                if submitted:
                    traditional[['Impressions']] = traditional[['Impressions']].fillna(
                        filldict[fill_blank_impressions_with])
                    traditional['Impressions'] = traditional['Impressions'].astype(int)
                    st.session_state.df_traditional = traditional
                    st.session_state.filled = True
                    st.experimental_rerun()



elif page == "5: Authors":
    st.title('Authors')
    traditional = st.session_state.df_traditional
    original_trad_auths = st.session_state.original_trad_auths

    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif len(traditional) == 0:
        st.subheader("No traditional media in data. Skip to next step.")
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
        counter = st.session_state.counter
        original_top_authors = st.session_state.original_auths

        headline_table = pd.pivot_table(traditional, index="Headline", values=["Mentions", "Author"], aggfunc="count")
        headline_table["Missing"] = headline_table["Mentions"] - headline_table["Author"]
        headline_table = headline_table[headline_table["Author"] > 0]
        headline_table = headline_table[headline_table['Missing'] > 0]
        headline_table = headline_table.sort_values("Missing", ascending=False)
        headline_table = headline_table.reset_index()
        headline_table.rename(columns={'Author': 'Known',
                           'Mentions': 'Total'},
                  inplace=True, errors='raise')

        temp_headline_list = headline_table
        if counter < len(temp_headline_list):
            headline_text = temp_headline_list.iloc[counter]['Headline']

            but1, col3, but2 = st.columns(3)
            with but1:
                next_auth = st.button('Skip to Next Headline')
                if next_auth:
                    counter += 1
                    st.session_state.counter = counter
                    st.experimental_rerun()

            if counter > 0:
                with col3:
                    st.write(f"Skipped: {counter}")
                with but2:
                    reset_counter = st.button('Reset Skip Counter')
                    if reset_counter:
                        counter = 0
                        st.session_state.counter = counter
                        st.experimental_rerun()


            possibles = headline_authors(traditional, headline_text)['index'].tolist()

            # CSS to inject contained in a string
            hide_table_row_index = """
                                <style>
                                tbody th {display:none}
                                .blank {display:none}
                                </style>
                                """
            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            st.table(headline_table.iloc[[counter]])
            st.table(headline_authors(traditional, headline_text).rename(columns={'index': 'Possible Author(s)',
                                                                                  'Author': 'Matches'}))

            with st.form('auth updater', clear_on_submit=True):

                box_author = st.selectbox('Pick from possible Authors', possibles, help='Pick from one of the authors already associated with this headline.')
                st.markdown("#### - OR -")
                string_author = st.text_input("Write in the author name", help='Override above selection by writing in a custom name.')

                if len(string_author) > 0:
                    new_author = string_author
                else:
                    new_author = box_author

                submitted = st.form_submit_button("Update Author")
                if submitted:
                    fix_author(traditional, headline_text, new_author)
                    st.session_state.df_traditional = traditional
                    st.experimental_rerun()
        else:
            st.write("You've reached the end of the list!")
            if counter > 0:
                reset_counter = st.button('Reset Counter')
                if reset_counter:
                    counter = 0
                    st.session_state.counter = counter
                    st.experimental_rerun()
            else:
                st.write("✓ Nothing left to update here.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Top Authors")
            st.write(original_trad_auths)


        with col2:
            st.subheader("New Top Authors")
            st.write(top_x_by_mentions(traditional, "Author"))


        st.subheader("Fixable Author Stats")
        stats = (fixable_headline_stats(traditional, primary="Headline", secondary="Author"))
        st.text(stats)


# TODO: Top Headlines / stories

elif page == "5.5: Author - Outlets":
    st.title("Author - Outlets")
    traditional = st.session_state.df_traditional
    auth_counter = st.session_state.auth_counter
    auth_outlet_table = st.session_state.auth_outlet_table
    from unidecode import unidecode

    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
        top_auths_by = st.selectbox('Top Authors by: ', ['Mentions', 'Impressions'])
        if len(auth_outlet_table) == 0:
            if top_auths_by == 'Mentions':
                auth_outlet_table = traditional[['Author', 'Mentions', 'Impressions']].groupby(by=['Author']).sum().sort_values(
                ['Mentions', 'Impressions'], ascending=False).reset_index()
            if top_auths_by == 'Impressions':
                auth_outlet_table = traditional[['Author', 'Mentions', 'Impressions']].groupby(
                    by=['Author']).sum().sort_values(
                    ['Impressions'], ascending=False).reset_index()

        auth_counter = st.session_state.auth_counter

        if auth_counter < len(auth_outlet_table):
            author_name = auth_outlet_table.iloc[auth_counter]['Author']
            # SKIP & RESET SKIP COUNTER SECTION

            # st.header(author_name.upper())
            # st.markdown("""
            #     <h2 style="color: goldenrod">
            #     """ + author_name +
            #             """</h2>""", unsafe_allow_html=True)

            # but1, col3, but2 = st.columns(3)
            # with but1:
            #     next_auth = st.button('Skip to Next Author')
            #     if next_auth:
            #         auth_counter += 1
            #         st.session_state.auth_counter = auth_counter
            #         st.experimental_rerun()
            #
            # if auth_counter > 0:
            #     with col3:
            #         st.write(f"Reviewed / Skipped: {auth_counter}")
            #     with but2:
            #         reset_counter = st.button('Reset Skips')
            #         if reset_counter:
            #             auth_counter = 0
            #             st.session_state.auth_counter = auth_counter
            #             st.experimental_rerun()

            col1, but1, but2 = st.columns([2,1,1])
            with col1:
                st.markdown("""
                                <h2 style="color: goldenrod; padding-top:0!important;"> 
                                """ + author_name.upper() +
                            """</h2>
                            <style>.css-12w0qpk {padding-top:15px !important}</style>
                            """, unsafe_allow_html=True)
            with but1:
                next_auth = st.button('Skip to Next Author')
                if next_auth:
                    auth_counter += 1
                    st.session_state.auth_counter = auth_counter
                    st.experimental_rerun()

                with but2:
                    reset_counter = st.button('Reset Skips')
                    if reset_counter:
                        auth_counter = 0
                        st.session_state.auth_counter = auth_counter
                        st.experimental_rerun()

            search_results = fetch_outlet(unidecode(author_name))
            number_of_authors = len(search_results['results'])
            db_outlets = []

            if search_results['results'] == []:
                matched_authors = []
            elif search_results == None:
                matched_authors = []
            else:
                response_results = search_results['results']
                outlet_results = []

                for result in response_results:
                    auth_name = result['firstName'] + " " + result['lastName']
                    job_title = result['primaryEmployment']['jobTitle']
                    outlet = result['primaryEmployment']['outletName']
                    country = result['country']['name']
                    auth_tuple = (auth_name, job_title, outlet, country)
                    outlet_results.append(auth_tuple)

                matched_authors = pd.DataFrame.from_records(outlet_results,
                                                            columns=['Name', 'Title', 'Outlet', 'Country'])
                matched_authors.loc[matched_authors.Outlet == "[Freelancer]", "Outlet"] = "Freelance"

                # coverage_outlets =
                db_outlets = matched_authors.Outlet.tolist()


            #####

            # OUTLETS IN COVERAGE VS DATABASE
            # CSS to inject contained in a string
            hide_table_row_index = """
                                            <style>
                                            tbody th {display:none}
                                            .blank {display:none}
                                            </style>
                                            """

            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {width:0; display:none}
                        .blank {width:0; display:none}
                        </style>
                        """

            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            # st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([8, 1, 16])
            with col1:
                st.subheader("Outlets in Coverage") #########################################
                outlets_in_coverage = traditional.loc[traditional.Author == author_name].Outlet.value_counts()
                outlets_in_coverage_list = outlets_in_coverage.index
                outlets_in_coverage_list = outlets_in_coverage_list.insert(0, "Freelance")

                # df = df.value_counts().rename_axis('unique_values').reset_index(name='counts')


                outlets_in_coverage = outlets_in_coverage.rename_axis('Outlet').reset_index(name='Matches')


                st.table(outlets_in_coverage.style.apply(
                    lambda x: ['background: goldenrod; color: black' if v in db_outlets else "" for v in x],
                    axis=1))

                # outlets_in_coverage = outlets_in_coverage.set_index('Outlet')
                #
                # st.dataframe(outlets_in_coverage.style.apply(
                #     lambda x: ['background: goldenrod; color: black' if v in db_outlets else "" for v in x],
                #     axis=1))



            with col2:
                st.write(" ")

            with col3:
                st.subheader("Media Database Matches") #####################################
                if search_results['results'] == []:
                    st.warning("NO MATCH FOUND")
                    matched_authors = []
                elif search_results == None:
                    st.warning("NO MATCH FOUND")
                    matched_authors = []
                else:
                    response_results = search_results['results']
                    outlet_results = []

                    for result in response_results:
                        auth_name = result['firstName'] + " " + result['lastName']
                        job_title = result['primaryEmployment']['jobTitle']
                        outlet = result['primaryEmployment']['outletName']
                        country = result['country']['name']
                        auth_tuple = (auth_name, job_title, outlet, country)
                        outlet_results.append(auth_tuple)
                        # print('///////////////////////')

                    matched_authors = pd.DataFrame.from_records(outlet_results, columns=['Name', 'Title', 'Outlet', 'Country'])
                    matched_authors.loc[matched_authors.Outlet == "[Freelancer]", "Outlet"] = "Freelance"


                    st.table(matched_authors.style.apply(lambda x: ['background: goldenrod; color: black' if v in outlets_in_coverage.Outlet.tolist() else "" for v in x], axis = 1))


                    possibles = matched_authors.Outlet



            # FORM TO UPDATE AUTHOR OUTLET ######################
            with st.form('auth updater', clear_on_submit=True):

                if len(matched_authors) > 0:
                    st.write('**DATABASE MATCHES FOUND!**')
                    box_outlet = st.selectbox('Pick from possible matches', possibles,
                                           help='Pick from one of the outlets associated with this author name.')

                else:
                    st.write('**NO DATABASE MATCH FOUND**')
                    box_outlet = st.selectbox('Pick "Freelance" or outlet from coverage', outlets_in_coverage_list)

                string_outlet = st.text_input("OR:  Write in the outlet name",
                                              help='Override above selection by writing in a custom name.')

                submitted = st.form_submit_button("Assign Outlet")

            if submitted:
                if len(string_outlet) > 0:
                    new_outlet = string_outlet
                else:
                    new_outlet = box_outlet

                auth_outlet_table.loc[auth_outlet_table["Author"] == author_name, "Outlet"] = new_outlet
                auth_counter += 1
                st.session_state.auth_counter = auth_counter
                st.session_state.auth_outlet_table = auth_outlet_table

                st.experimental_rerun()



            col1, col2, col3 = st.columns([6, 1, 4])
            with col1:
                st.subheader("Top Authors")
                if 'Outlet' in auth_outlet_table.columns:
                    if top_auths_by == 'Mentions':
                        st.table(auth_outlet_table[['Author', 'Outlet', 'Mentions', 'Impressions']].fillna('').sort_values(
                            ['Mentions', 'Impressions'], ascending=False).head(15).style.format(format_dict))
                    if top_auths_by == 'Impressions':
                        st.table(auth_outlet_table[['Author', 'Outlet', 'Mentions', 'Impressions']].fillna('').sort_values(
                            ['Impressions', 'Mentions'], ascending=False).head(15).style.format(format_dict))


                else:
                    st.table(auth_outlet_table[['Author', 'Mentions', 'Impressions']].fillna('').head(15).style.format(
                        format_dict))
            with col2:
                st.write(" ")
            with col3:
                st.subheader('Outlets assigned')
                st.metric(label='Assigned', value=auth_outlet_table.Outlet.count())
        else:
            st.write("You've reached the end of the list!")
            if auth_counter > 0:
                reset_counter = st.button('Reset Counter')
                if reset_counter:
                    auth_counter = 0
                    st.session_state.auth_counter = auth_counter
                    st.experimental_rerun()
            else:
                st.write("✓ Nothing left to update here.")


        # TODO: if table exists, table to sheet on export
        # TODO: layout improvements?
        # TODO: (maybe later) toggle to choose look at top Authors by impressions or mentions
        # TODO: NICE TO HAVE: job title for database matched authors


elif page == "6: Translation":
    st.title('Translation')
    traditional = st.session_state.df_traditional
    social = st.session_state.df_social
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    elif st.session_state.translated_headline == True and st.session_state.translated_snippet == True and st.session_state.translated_summary == True:
        st.subheader("✓ Translation complete.")
    elif len(traditional[traditional['Language'] != 'English']) == 0 and len(social[social['Language'] != 'English']) == 0:
        st.subheader("No translation required")
    else:
        translation_stats_combo()
        st.write(len(traditional[traditional['Language'] != 'English']))
        st.write(len(social[social['Language'] != 'English']))
        if len(traditional) > 0:
            with st.expander("Traditional - Non-English"):
                st.dataframe(traditional[traditional['Language'] != 'English'][['Outlet', 'Headline','Snippet','Summary','Language','Country']])

        if len(social) > 0:
            with st.expander("Social - Non-English"):
                st.dataframe(social[social['Language'] != 'English'][
                                 ['Outlet', 'Snippet', 'Summary', 'Language', 'Country']])

        with st.form('translation_form'):
            st.subheader("Pick columns for translations")
            st.warning("WARNING: Translation will over-write the original text.")

            if len(traditional) > 0:
                if st.session_state.translated_headline == False:
                    headline_to_english = st.checkbox('Headline')
                else:
                    st.success('✓ Headlines translated.')
                    headline_to_english = False
            else:
                headline_to_english = False

            if st.session_state.translated_snippet == False:
                snippet_to_english = st.checkbox('Snippet')
            else:
                st.success('✓ Snippets translated.')
                snippet_to_english = False

            if st.session_state.translated_summary == False:
                summary_to_english = st.checkbox('Summary')
            else:
                st.success('✓ Summaries translated.')
                summary_to_english = False

            submitted = st.form_submit_button("Go!")
            if submitted:
                st.warning("Stay on this page until translation is complete")

                if headline_to_english:
                    translate_col(traditional, 'Headline')

                    # AP Cap
                    broadcast_array = ['RADIO', 'TV']
                    broadcast = traditional.loc[traditional['Type'].isin(broadcast_array)]
                    traditional = traditional[~traditional['Type'].isin(broadcast_array)]
                    traditional[['Headline']] = traditional[['Headline']].fillna('')
                    traditional['Headline'] = traditional['Headline'].map(lambda Headline: titlecase(Headline))
                    frames = [traditional, broadcast]
                    traditional = pd.concat(frames)

                    translate_col(social, 'Headline')
                    st.session_state.translated_headline = True
                    st.success(f'Done translating headlines!')
                if summary_to_english:
                    translate_col(traditional, 'Summary')
                    translate_col(social, 'Summary')
                    st.session_state.translated_summary = True
                    st.success(f'Done translating summaries!')
                if snippet_to_english:
                    translate_col(traditional, 'Snippet')
                    translate_col(social, 'Snippet')
                    st.session_state.translated_snippet = True
                    st.success(f'Done translating snippets!')
                st.session_state.df_traditional = traditional
                st.session_state.df_social = social
                st.experimental_rerun()


elif page == "7: Review":
    st.title('Review')
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
        traditional = st.session_state.df_traditional
        social = st.session_state.df_social
        dupes = st.session_state.df_dupes

        if len(traditional) > 0:
            with st.expander("Traditional"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Basic Metrics")
                    st.metric(label="Mentions", value="{:,}".format(len(traditional)))
                    st.metric(label="Impressions", value="{:,}".format(traditional['Impressions'].sum()))
                with col2:
                    st.subheader("Media Type")
                    st.write(traditional['Type'].value_counts())

                col3, col4 = st.columns(2)
                with col3:
                    st.subheader("Top Authors")
                    top_authors = (top_x_by_mentions(traditional, "Author"))
                    st.table(top_authors)

                with col4:
                    st.subheader("Top Outlets")
                    top_outlets = (top_x_by_mentions(traditional, "Outlet"))
                    st.table(top_outlets)


                st.markdown('##')
                st.subheader('Mention Trend')

                trend = alt.Chart(traditional).mark_line().encode(
                    x='Date:T',
                    y='count(Mentions):Q'
                )
                st.altair_chart(trend, use_container_width=True)

                st.markdown('##')
                st.subheader('Impressions Trend')

                trend2 = alt.Chart(traditional).mark_line().encode(
                    x='Date:T',
                    y='sum(Impressions):Q'
                )
                st.altair_chart(trend2, use_container_width=True)

                st.subheader("Cleaned Data")
                st.dataframe(traditional.style.format(format_dict))
                st.markdown('##')

        if len(social) > 0:
            with st.expander("Social"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Basic Metrics")
                    st.metric(label="Mentions", value="{:,}".format(len(social)))
                    st.metric(label="Impressions", value="{:,}".format(social['Impressions'].sum()))
                with col2:
                    st.subheader("Media Type")
                    st.write(social['Type'].value_counts())

                # col3, col4 = st.columns(2)
                # with col3:
                #     st.subheader("Top Authors")
                #     top_authors = (top_x_by_mentions(social, "Author"))
                #     st.write(top_authors)
                #
                # with col4:
                #     st.subheader("Top Outlets")
                #     top_outlets = (top_x_by_mentions(social, "Outlet"))
                #     st.write(top_outlets)

                st.markdown('##')
                st.subheader('Mention Trend')

                trend = alt.Chart(social).mark_line().encode(
                    x='Date:T',
                    y='count(Mentions):Q'
                )
                st.altair_chart(trend, use_container_width=True)

                st.markdown('##')
                st.subheader('Impressions Trend')

                trend2 = alt.Chart(social).mark_line().encode(
                    x='Date:T',
                    y='sum(Impressions):Q'
                )
                st.altair_chart(trend2, use_container_width=True)

                st.subheader("Cleaned Data")
                st.dataframe(social.style.format(format_dict))
                st.markdown('##')

        if len(dupes) > 0:
            with st.expander("Duplicates Removed"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Basic Metrics")
                    st.metric(label="Mentions", value="{:,}".format(len(dupes)))
                    st.metric(label="Impressions", value="{:,}".format(dupes['Impressions'].sum()))
                with col2:
                    st.subheader("Media Type")
                    st.write(dupes['Type'].value_counts())
                st.dataframe(dupes.style.format(format_dict))


elif page == "8: Download":
    # TODO: ONLY INCLUDE CLEAN SHEETS FOR data sets that exist
    st.title('Download')
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
        traditional = st.session_state.df_traditional
        social = st.session_state.df_social
        dupes = st.session_state.df_dupes
        uncleaned = st.session_state.df_uncleaned
        export_name = st.session_state.export_name
        auth_outlet_table = st.session_state.auth_outlet_table

        traditional['Date'] = pd.to_datetime(traditional['Date'])
        social['Date'] = pd.to_datetime(social['Date'])
        dupes['Date'] = pd.to_datetime(dupes['Date'])

        with st.form("my_form_download"):
            st.subheader("Generate your cleaned data workbook")
            submitted = st.form_submit_button("Go!")
            if submitted:
                with st.spinner('Building workbook now...'):

                    traditional = traditional.sort_values(by=['Impressions'], ascending=False)
                    social = social.sort_values(by=['Impressions'], ascending=False)
                    authors = auth_outlet_table.sort_values(by=['Mentions', 'Impressions'], ascending=False)

                    output = io.BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd',
                                            options={'in_memory': True})

                    # Write the dataframe data to XlsxWriter.
                    traditional.to_excel(writer, sheet_name='CLEAN TRAD', startrow=1, header=False, index=False)
                    social.to_excel(writer, sheet_name='CLEAN SOCIAL', startrow=1, header=False, index=False)
                    # if len(authors) > 0:
                    authors.to_excel(writer, sheet_name='Authors', header=True, index=False)
                    dupes.to_excel(writer, sheet_name='DLTD DUPES', header=True, index=False)
                    uncleaned.to_excel(writer, sheet_name='RAW', header=True, index=False)

                    # Get the xlsxwriter workbook and worksheet objects.
                    workbook = writer.book
                    worksheet1 = writer.sheets['CLEAN TRAD']
                    worksheet2 = writer.sheets['CLEAN SOCIAL']
                    worksheet3 = writer.sheets['DLTD DUPES']
                    worksheet4 = writer.sheets['RAW']
                    worksheet5 = writer.sheets['Authors']

                    worksheet1.set_tab_color('black')
                    worksheet2.set_tab_color('black')
                    worksheet3.set_tab_color('#c26f4f')
                    worksheet4.set_tab_color('#c26f4f')
                    worksheet5.set_tab_color('green')

                    # make a list of df/worksheet tuples
                    cleaned_dfs = [(traditional, worksheet1), (social, worksheet2), (dupes, worksheet3)]

                    for clean_df in cleaned_dfs:
                        (max_row, max_col) = clean_df[0].shape
                        column_settings = [{'header': column} for column in clean_df[0].columns]
                        clean_df[1].add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

                    # make a list of cleaned worksheets
                    cleaned_sheets = [worksheet1, worksheet2, worksheet3]

                    # Add some cell formats.
                    number_format = workbook.add_format({'num_format': '#,##0'})
                    currency_format = workbook.add_format({'num_format': '$#,##0'})
                    # date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
                    time_format = workbook.add_format({'num_format': 'hh:mm:ss'})

                    # Add the Excel table structure. Pandas will add the data.
                    for sheet in cleaned_sheets:
                        sheet.set_default_row(22)
                        sheet.set_column('A:A', 12, None)  # date
                        sheet.set_column('B:B', 10, time_format)  # time
                        sheet.set_column('C:C', 22, None)  # outlet
                        sheet.set_column('D:D', 12, None)  # type
                        sheet.set_column('E:E', 12, None)  # author
                        sheet.set_column('F:F', 0, None)  # mentions
                        sheet.set_column('G:G', 12, number_format)  # impressions
                        sheet.set_column('H:H', 40, None)  # headline
                        sheet.set_column('R:R', 12, currency_format)  # AVE
                        sheet.freeze_panes(1, 0)
                    writer.save()

        if submitted:
            st.download_button('Download', output, file_name=export_name)