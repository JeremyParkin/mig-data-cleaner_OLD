import streamlit as st
import pandas as pd
import numpy as np
import io
from io import BytesIO
import xlsxwriter
from deep_translator import GoogleTranslator
from titlecase import titlecase
import warnings
from datetime import datetime
import plotly.graph_objects as go
import altair as alt
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

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
    unique_non_eng = list(set(df[name_of_column][df['Language'] != 'English'].dropna()))
    if '' in unique_non_eng:
        unique_non_eng.remove('')
    unique_non_eng_clipped = []
    with st.spinner('Running translation now...'):
        for text in unique_non_eng:
            unique_non_eng_clipped.append(text[:1500])
        translated_x = []
        for text in unique_non_eng_clipped:
            translated_x.append(GoogleTranslator(source='auto', target='en').translate(text))
            dictionary = dict(zip(unique_non_eng, translated_x))
            df[name_of_column].replace(dictionary, inplace = True)
    st.success('Done!')


def translation_stats_combo():
    non_english_records = len(traditional[traditional['Language'] != 'English']) + len(social[social['Language'] != 'English'])
    minutes = non_english_records//100
    if minutes == 0:
        min_word = 'minute'
    else:
        min_word = 'minutes'
    st.write(f"There are {non_english_records} non-English records in your data.")
    st.write(f"\nAllow around {minutes}-{minutes + 1} {min_word} per column for translation.")


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


# Sidebar page selector
st.sidebar.image('https://agilitypr.news/images/Agility-centered.svg', width=200)
st.sidebar.title('MIG: Data Cleaning App')
page = st.sidebar.radio("Data Cleaning Steps:", [
    "1: Upload your CSV",
    "2: Standard Cleaning",
    "3: Impressions - Outliers",
    "4: Impressions - Fill Blanks",
    "5: Authors",
    "6: Translation",
    "7: Download"], index=0)

if page == "1: Upload your CSV":
    st.session_state['page'] = '1: Upload your CSV'
    st.header('Getting Started')

    with st.form("my_form"):
        client = st.text_input('Client organization name*', placeholder='eg. Air Canada', key='client')
        period = st.text_input('Reporting period*', placeholder='eg. March 2022', key='period')
        uploaded_file = st.file_uploader(label='Upload your CSV*', type='csv',
                                         accept_multiple_files=False)
        submitted = st.form_submit_button("Submit")
        if submitted and (client == "" or period == "" or uploaded_file == None):
            st.error('Missing required form inputs above.')

        elif submitted:
            data = pd.read_csv(uploaded_file)
            data = data.dropna(thresh=2)
            st.session_state.df_uncleaned = data
            st.session_state.df_raw = data
            st.session_state.upload_step = True
            data["Mentions"] = 1
            data['Audience Reach'] = data['Audience Reach'].astype('Int64')
            st.session_state.export_name = f"{client} - {period} - clean_data.xlsx"
            st.session_state.page_subtitle = f"{client} - {period}"

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Basic Metrics")
                st.metric(label="Mentions", value="{:,}".format(len(data)))
                st.metric(label="Impressions", value="{:,}".format(data['Audience Reach'].sum()))
                # st.metric(label="AVE", value="{:,}".format(data['AVE'].sum()))
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

            st.markdown('##')
            st.subheader('Trend')
            # data['Published Date'] = pd.to_datetime(data['Published Date'])

            trend = alt.Chart(data).mark_line().encode(
                x='Published Date:T',
                y='count(Mentions):Q'
            )
            st.altair_chart(trend, use_container_width=True)

            st.subheader("Raw Data")
            st.dataframe(data)
            st.markdown('##')

            st.subheader("Data Stats")
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)


elif page == "2: Standard Cleaning":
    st.header('Standard Cleaning')
    st.markdown(st.session_state.page_subtitle)
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    else:
        data = st.session_state.df_raw
        with st.form("my_form_basic_cleaning"):
            st.subheader("Run standard cleaning")
            submitted = st.form_submit_button("Go!")
            if submitted:
                data = data.rename(columns={
                    'Published Date': 'Date',
                    'Published Time': 'Time',
                    'Media Type': 'Type',
                    'Coverage Snippet': 'Snippet',
                    'Province/State': 'Prov/State',
                    'Audience Reach': 'Impressions'})
                st.write('✓ Columns Renamed')

                data.Type.replace({"ONLINE_NEWS": "ONLINE NEWS", "PRESS_RELEASE": "PRESS RELEASE"}, inplace=True)
                data.loc[data['URL'].str.contains("www.facebook.com", na=False), 'Type'] = "FACEBOOK"
                data.loc[data['URL'].str.contains("/twitter.com", na=False), 'Type'] = "TWITTER"
                data.loc[data['URL'].str.contains("www.instagram.com", na=False), 'Type'] = "INSTAGRAM"
                data.loc[data['URL'].str.contains("reddit.com", na=False), 'Type'] = "REDDIT"
                data.loc[data['URL'].str.contains("youtube.com", na=False), 'Type'] = "YOUTUBE"
                st.write('✓ Media Types Cleaned')

                if "Original URL" in data:
                    data.loc[data["Original URL"].notnull(), "URL"] = data["Original URL"]
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
                st.write('✓ Useless Columns Dropped')

                # Move columns
                temp = data.pop('Impressions')
                data.insert(5, 'Impressions', temp)
                # data.Impressions = data.Impressions.astype('Int64')
                temp = data.pop('Mentions')
                data.insert(5, 'Mentions', temp)

                # Strip extra white space
                data['Headline'].str.strip()
                data['Outlet'].str.strip()
                data['Author'].str.strip()
                data['Headline'] = data['Headline'].str.replace('   ', ' ')
                data['Outlet'] = data['Outlet'].str.replace('   ', ' ')
                data['Author'] = data['Author'].str.replace('   ', ' ')
                data['Headline'] = data['Headline'].str.replace('  ', ' ')
                data['Outlet'] = data['Outlet'].str.replace('  ', ' ')
                data['Author'] = data['Author'].str.replace('  ', ' ')
                st.write('✓ Extra Spaces Removed')

                # Remove (Online)
                data['Outlet'] = data['Outlet'].str.replace(' \(Online\)', '')
                st.write('✓ "(Online)" Removed from Outlet Names')

                # Tag exploder
                if "Tags" in data:
                    data = data.join(data["Tags"].str.get_dummies(sep=","))
                    st.write('✓ Tags Expanded to Unique Columns')

                # DROP SOCIALS To sep df
                soc_array = ['FACEBOOK', 'TWITTER', 'INSTAGRAM', 'REDDIT', 'YOUTUBE']
                social = data.loc[data['Type'].isin(soc_array)]
                index_names = data[(data['Type'] == 'FACEBOOK')].index
                data.drop(index_names, inplace=True)
                index_names = data[(data['Type'] == 'TWITTER')].index
                data.drop(index_names, inplace=True)
                index_names = data[(data['Type'] == 'INSTAGRAM')].index
                data.drop(index_names, inplace=True)
                index_names = data[(data['Type'] == 'REDDIT')].index
                data.drop(index_names, inplace=True)
                index_names = data[(data['Type'] == 'YOUTUBE')].index
                data.drop(index_names, inplace=True)
                index_names = data[(data['Type'] == 'PODCAST')].index
                data.drop(index_names, inplace=True)
                st.write('✓ Social Split Out')

                # AP Cap
                broadcast_array = ['RADIO', 'TV']
                broadcast = data.loc[data['Type'].isin(broadcast_array)]

                index_names = data[(data['Type'] == 'RADIO')].index
                data.drop(index_names, inplace=True)
                index_names = data[(data['Type'] == 'TV')].index
                data.drop(index_names, inplace=True)

                data[['Headline']] = data[['Headline']].fillna('')
                data['Headline'] = data['Headline'].map(lambda Headline: titlecase(Headline))

                frames = [data, broadcast]
                data = pd.concat(frames)
                st.write('✓ AP Style Capitalization')

                # Yahoo standardizer
                yahoo_cleanup('sports.yahoo.com')
                yahoo_cleanup('www.yahoo.com')
                yahoo_cleanup('news.yahoo.com')
                yahoo_cleanup('style.yahoo.com')
                yahoo_cleanup('finance.yahoo.com')
                st.write('✓ Yahoo Standardization')

                # Drop dupes
                broadcast_array = ['RADIO', 'TV']
                broadcast = data.loc[data['Type'].isin(broadcast_array)]

                # DROP BROADCAST
                index_names = data[(data['Type'] == 'RADIO')].index
                data.drop(index_names, inplace=True)
                index_names = data[(data['Type'] == 'TV')].index
                data.drop(index_names, inplace=True)

                # Add temporary dupe URL helper column
                data['URL Helper'] = data['URL'].str.lower()
                data['URL Helper'] = data['URL Helper'].str.replace('http:', 'https:')

                # Save duplicate URLS
                dupe_urls = data[data['URL Helper'].duplicated(keep='first') == True]
                # Drop duplicate URLs
                data = data[
                    data['URL Helper'].isnull() | ~data[data['URL Helper'].notnull()].duplicated(subset='URL Helper',
                                                                                                 keep='first')]
                # Drop URL Helper column
                data.drop(["URL Helper"], axis=1, inplace=True, errors='ignore')
                dupe_urls.drop(["URL Helper"], axis=1, inplace=True, errors='ignore')
                # SAVE duplicates based on TYPE + OUTLET + HEADLINE
                data["dupe_helper"] = data['Type'] + data['Outlet'] + data['Headline']
                dupe_cols = data[data['dupe_helper'].duplicated(keep='first') == True]
                # Drop other duplicates based on TYPE + OUTLET + HEADLINE
                data.drop(["dupe_helper"], axis=1, inplace=True, errors='ignore')
                dupe_cols.drop(["dupe_helper"], axis=1, inplace=True, errors='ignore')

                frames = [data, broadcast]
                data = pd.concat(frames)
                dupes = pd.concat([dupe_urls, dupe_cols])
                st.write('✓ Duplicates Removed')

                if len(data) > 0:
                    st.subheader("Traditional")
                    st.dataframe(data)
                if len(social) > 0:
                    st.subheader("Social")
                    st.dataframe(social)
                if len(dupes) > 0:
                    st.subheader("Deleted Dupes")
                    st.dataframe(dupes)

                st.session_state.df_traditional = data
                st.session_state.df_social = social
                st.session_state.df_dupes = dupes

                st.session_state.standard_step = True

                # Dual axis trend chart
                # base = alt.Chart(data).encode(x='Date:T')
                # bar = base.mark_bar(size=20).encode(y='count(Mentions):Q')
                # line = base.mark_line(color='red').encode(y='Impressions:Q')
                # trend3 = alt.layer(bar, line).resolve_scale(y='independent')
                # st.altair_chart(trend3, use_container_width=True)


elif page == "3: Impressions - Outliers":
    traditional = st.session_state.df_traditional
    st.header('Impressions - Outliers')
    st.markdown(st.session_state.page_subtitle)
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
        st.subheader('Check highest impressions numbers:')
        st.dataframe(traditional[['Outlet', 'Type', 'Impressions', 'Headline', 'URL', 'Country']].nlargest(25, 'Impressions'))

        with st.form("Update Outliers"):
            st.subheader("Update Impressions Outliers")
            index_number = int(st.number_input('Row index number: ', step=1, format='%i'))
            new_impressions_value = int(st.number_input('New impressions value for row', step=1, format='%i'))
            submitted = st.form_submit_button("Go!")
            if submitted:
                traditional.loc[index_number, "Impressions"] = new_impressions_value
                st.session_state.df_traditional = traditional
                st.session_state.outliers = True
                st.experimental_rerun()

        if st.session_state.outliers == False:
            done_outliers = st.button('Done with outliers')
            if done_outliers:
                st.session_state.outliers = True

elif page == "4: Impressions - Fill Blanks":
    st.header('Impressions - Fill Blanks')
    st.markdown(st.session_state.page_subtitle)
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    elif st.session_state.outliers == False:
        st.error('Please confirm outliers step is complete before running this step.')
        done_outliers = st.button('Done with outliers')
        if done_outliers:
            st.session_state.outliers = True
            st.experimental_rerun()

    else:
        # st.subheader('Check statistical levels')
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

        st.write(f"\n*************\nMISSING: {blank_impressions}\n*************\n")

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
                fill_blank_impressions_with = st.radio('Pick your statistical fill value: ', filldict.keys(), index=4)
                submitted = st.form_submit_button("Fill Blanks")
                if submitted:
                    traditional[['Impressions']] = traditional[['Impressions']].fillna(
                        filldict[fill_blank_impressions_with])
                    traditional['Impressions'] = traditional['Impressions'].astype(int)
                    st.session_state.df_traditional = traditional
                    st.experimental_rerun()


elif page == "5: Authors":
    st.header('Authors')
    st.markdown(st.session_state.page_subtitle)
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
        traditional = st.session_state.df_traditional
        counter = st.session_state.counter
        original_top_authors = st.session_state.original_auths

        headline_table = pd.pivot_table(traditional, index="Headline", values=["Mentions", "Author"], aggfunc="count")
        headline_table["Missing"] = headline_table["Mentions"] - headline_table["Author"]
        headline_table = headline_table[headline_table["Author"] > 0]
        headline_table = headline_table[headline_table['Missing'] > 0]
        headline_table = headline_table.sort_values("Missing", ascending=False)
        headline_table = headline_table.reset_index()

        temp_headline_list = headline_table
        if counter < len(temp_headline_list):
            headline_text = temp_headline_list.iloc[counter]['Headline']

            st.markdown("#### Fixable Headline")
            st.text(headline_text)

            but1, but2 = st.columns(2)
            with but1:
                next_auth = st.button('Skip to Next Headline')
                if next_auth:
                    counter += 1
                    st.session_state.counter = counter
                    st.experimental_rerun()

            if counter > 0:
                with but2:
                    reset_counter = st.button('Reset Counter')
                    if reset_counter:
                        counter = 0
                        st.session_state.counter = counter
                        st.experimental_rerun()

            possibles = headline_authors(traditional, headline_text)['index'].tolist()
            possibles.append('- other - ')

            # st.write(f"Counter: {counter}")
            # st.write(f"Length of table: {len(temp_headline_list)}")

            with st.form('auth updater', clear_on_submit=True):
                st.write("**Possible Authors**")
                st.write(headline_authors(traditional, headline_text))
                box_author = st.selectbox('Possible Authors', possibles)
                string_author = st.text_input("What name should be applied to the author field?", help='This will over-ride a selection from the dropdown above.')

                if len(string_author) > 0:
                    new_author = string_author
                else:
                    new_author = box_author

                submitted = st.form_submit_button("Update Author")
                if submitted:
                    if new_author == '- other -':
                        st.error("Oops! Try again.")
                    else:
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


        st.subheader('Most fixable headline authors')

        st.table(headline_table.head(10))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Top Authors")
            st.write(original_top_authors)

        with col2:
            st.subheader("New Top Authors")
            st.write(top_x_by_mentions(traditional, "Author"))

        st.subheader("Fixable Author Stats")
        stats = (fixable_headline_stats(traditional, primary="Headline", secondary="Author"))
        st.text(stats)


elif page == "6: Translation":
    st.header('Translation')
    st.markdown(st.session_state.page_subtitle)
    traditional = st.session_state.df_traditional
    social = st.session_state.df_social
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    elif st.session_state.translated_headline == True and st.session_state.translated_snippet == True and st.session_state.translated_summary == True:
        st.subheader("✓ Translation complete.")
    elif len(traditional[traditional['Language'] != 'English']) == 0:
        st.subheader("No translation required")
    else:
        translation_stats_combo()
        st.dataframe(traditional[traditional['Language'] != 'English'][['Outlet', 'Headline','Snippet','Summary','Language','Country']])

        with st.form('translation_form'):
            st.subheader("Pick columns for translations")

            if st.session_state.translated_headline == False:
                headline_to_english = st.checkbox('Headline')
            else:
                st.write('✓ Headlines already translated.')
                headline_to_english = False

            if st.session_state.translated_snippet == False:
                snippet_to_english = st.checkbox('Snippet')
            else:
                st.write('✓ Snippets already translated.')
                snippet_to_english = False

            if st.session_state.translated_summary == False:
                summary_to_english = st.checkbox('Summary')
            else:
                st.write('✓ Summaries already translated.')
                summary_to_english = False

            submitted = st.form_submit_button("Go!")
            if submitted:
                if headline_to_english:
                    translate_col(traditional, 'Headline')
                    translate_col(social, 'Headline')
                    st.session_state.translated_headline = True
                if summary_to_english:
                    translate_col(traditional, 'Summary')
                    translate_col(social, 'Summary')
                    st.session_state.translated_summary = True
                if snippet_to_english:
                    translate_col(traditional, 'Snippet')
                    translate_col(social, 'Snippet')
                    st.session_state.translated_snippet = True
                st.session_state.df_traditional = traditional
                st.session_state.df_social = social
                st.experimental_rerun()



elif page == "7: Download":
    st.header('Download')
    st.markdown(st.session_state.page_subtitle)

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

                    output = io.BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd',
                                            options={'in_memory': True})

                    # Write the dataframe data to XlsxWriter.
                    traditional.to_excel(writer, sheet_name='CLEAN TRAD', startrow=1, header=False, index=False)
                    social.to_excel(writer, sheet_name='CLEAN SOCIAL', startrow=1, header=False, index=False)
                    dupes.to_excel(writer, sheet_name='DLTD DUPES', header=True, index=False)
                    uncleaned.to_excel(writer, sheet_name='RAW', header=True, index=False)

                    # Get the xlsxwriter workbook and worksheet objects.
                    workbook = writer.book
                    worksheet1 = writer.sheets['CLEAN TRAD']
                    worksheet2 = writer.sheets['CLEAN SOCIAL']
                    worksheet3 = writer.sheets['DLTD DUPES']
                    worksheet4 = writer.sheets['RAW']

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

                    # Close the Pandas Excel writer and output the Excel file.
                    writer.save()

        if submitted:
            st.download_button('Download', output, file_name=export_name)