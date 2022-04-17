import streamlit as st
import pandas as pd
import numpy as np

# from io import BytesIO
# import xlsxwriter

import warnings
# from threading import Thread


warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="MIG Data Cleaning App",
                   page_icon="https://www.agilitypr.com/wp-content/uploads/2018/02/favicon-192.png")
hide_menu_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

def top_x_by_mentions(df, column_name):
    """Returns top N items by mention count"""
    x_table = pd.pivot_table(df, index=column_name, values=["Mentions"], aggfunc="count")
    x_table = x_table.sort_values("Mentions", ascending=False)
    x_table = x_table.rename(columns={"Mentions": "Hits"})
    return x_table.head(10)


format_dict = {'AVE': '${0:,.0f}', 'Audience Reach': '{:,d}', 'Impressions': '{:,d}'}

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
if 'auth_outlet_skipped' not in st.session_state:
    st.session_state.auth_outlet_skipped = 0
if 'auth_outlet_table' not in st.session_state:
    st.session_state.auth_outlet_table = pd.DataFrame()
if 'top_auths_by' not in st.session_state:
    st.session_state.top_auths_by = 'Mentions'

# Sidebar and page selector
st.sidebar.image('https://agilitypr.news/images/Agility-centered.svg', width=200)
st.sidebar.title('MIG: Data Cleaning App')
pagelist = [
    "1: Getting Started",
    "2: Standard Cleaning",
    "3: Impressions - Outliers",
    "4: Impressions - Fill Blanks",
    "5: Authors - Missing",
    "6: Authors - Outlets",
    "7: Translation",
    "8: Review",
    "9: Download"]

page = st.sidebar.radio("Data Cleaning Steps:", pagelist, index=0)
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.markdown("")
st.sidebar.write("**Quick Links**")
st.sidebar.markdown("""
    [Bug Tracker](https://mediamiser.sharepoint.com/:x:/s/MIG/ESIF0YBKmq9EuXffFemDJKsBOuTn05Wii9ABFcj_q39A0A?e=zZ9oB2) \n
    [Quickstart Guide](https://github.com/JeremyParkin/mig-data-cleaner/blob/main/README.md) \n
    [GitHub Project](https://github.com/JeremyParkin/mig-data-cleaner) 
    """)
st.sidebar.caption("v.1.5.2.1")


if page == "1: Getting Started":
    st.title('Getting Started')
    import altair as alt
    import io

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


        st.markdown('##')


        st.subheader('Mention Trend')

        trend = data.groupby('Published Date')[['Mentions']].sum()
        # trend.plot()
        st.line_chart(trend, use_container_width=True)

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
        st.markdown('(First 50 rows)')
        st.dataframe(data.head(50).style.format(format_dict))
        st.markdown('##')

        with st.expander('Data set stats'):
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

    else:
        with st.form("my_form"):
            client = st.text_input('Client organization name*', placeholder='eg. Air Canada', key='client',
                                   help='Required to build export file name.')
            period = st.text_input('Reporting period or focus*', placeholder='eg. March 2022', key='period',
                                   help='Required to build export file name.')
            uploaded_file = st.file_uploader(label='Upload your CSV*', type='csv',
                                             accept_multiple_files=False,
                                             help='Only use CSV files exported from the Agility Platform.')

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
    from titlecase import titlecase

    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step:
        st.success("Standard cleaning already done!")
        traditional = st.session_state.df_traditional
        social = st.session_state.df_social
        dupes = st.session_state.df_dupes
        if len(traditional) > 0:
            with st.expander("Traditional"):
                st.markdown('(First 50 rows)')
                st.dataframe(traditional.head(50).style.format(format_dict))
        if len(social) > 0:
            with st.expander("Social"):
                st.markdown('(First 50 rows)')
                st.dataframe(social.head(50).style.format(format_dict))
        if len(dupes) > 0:
            with st.expander("Deleted Duplicates"):
                st.dataframe(dupes.style.format(format_dict))
    else:
        def yahoo_cleanup(url_string):
            data.loc[data['URL'].str.contains(url_string, na=False), "Outlet"] = "Yahoo! News"
            data.loc[data['URL'].str.contains(url_string, na=False), "Impressions"] = 80828000
            data.loc[data['URL'].str.contains(url_string, na=False), "Country"] = (np.nan)
            data.loc[data['URL'].str.contains(url_string, na=False), "Continent"] = (np.nan)
            data.loc[data['URL'].str.contains(url_string, na=False), "City"] = (np.nan)
            data.loc[data['URL'].str.contains(url_string, na=False), "Prov/State"] = (np.nan)
            data.loc[data['URL'].str.contains(url_string, na=False), "sub"] = data['URL'].str.rsplit('/', 1).str[-1]
            data.loc[data['URL'].str.contains(url_string, na=False), "URL"] = 'https://news.yahoo.com/' + data["sub"]
            data.drop(["sub"], axis=1, inplace=True, errors='ignore')


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


        data = st.session_state.df_raw

        with st.form("my_form_basic_cleaning"):
            st.subheader("Cleaning options")
            merge_online = st.checkbox("Merge 'blogs' and 'press releases' into 'Online'", value=True)
            fill_known_imp = st.checkbox("Fill missing impressions values where known match exists in data", value=True)
            submitted = st.form_submit_button("Go!")
            if submitted:
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
                        
                    data.Type.replace({"ONLINE_NEWS": "ONLINE NEWS", "PRESS_RELEASE": "PRESS RELEASE"}, inplace=True)
                    data.loc[data['URL'].str.contains("www.facebook.com", na=False), 'Type'] = "FACEBOOK"
                    data.loc[data['URL'].str.contains("/twitter.com", na=False), 'Type'] = "TWITTER"
                    data.loc[data['URL'].str.contains("www.instagram.com", na=False), 'Type'] = "INSTAGRAM"
                    data.loc[data['URL'].str.contains("reddit.com", na=False), 'Type'] = "REDDIT"
                    data.loc[data['URL'].str.contains("youtube.com", na=False), 'Type'] = "YOUTUBE"

                    if merge_online:
                        data.Type.replace({
                            "ONLINE NEWS": "ONLINE",
                            "PRESS RELEASE": "ONLINE",
                            "BLOGS": "ONLINE"}, inplace=True)

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
                        data['Tags'] = data['Tags'].astype(str)  # needed if column there but all blank
                        data = data.join(data["Tags"].str.get_dummies(sep=","))
                        with col2:
                            st.write('✓ Tags Expanded')

                    data = data.astype(
                        {"Type": 'category', "Sentiment": 'category', "Continent": 'category',
                         "Country": 'category', "Prov/State": 'category', "City": 'category', "Language": 'category'})

                    # SOCIALS To sep df
                    soc_array = ['FACEBOOK', 'TWITTER', 'INSTAGRAM', 'REDDIT', 'YOUTUBE']
                    social = data.loc[data['Type'].isin(soc_array)]
                    data = data[~data['Type'].isin(soc_array)]

                    with col3:
                        st.write('✓ Social Split Out')

                    # Fill known impressions
                    if fill_known_imp:
                        imp_fix_table = fixable_impressions_list(data)
                        for outlet in imp_fix_table.Outlet:
                            if len(outlet_imp(data, outlet)) == 1:
                                fix_imp(data, outlet, int(outlet_imp(data, outlet)['index']))

                    # AP Cap
                    broadcast_array = ['RADIO', 'TV']
                    broadcast = data.loc[data['Type'].isin(broadcast_array)]
                    data = data[~data['Type'].isin(broadcast_array)]

                    data[['Headline']] = data[['Headline']].fillna('')
                    data['Headline'] = data['Headline'].map(lambda Headline: titlecase(Headline))
                    # frames = [data, broadcast]
                    # data = pd.concat(frames)
                    with col1:
                        st.write('✓ AP Style Capitalization')

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

                    # Sort duplicate URLS
                    data = data.sort_values(["URL_Helper", "Author", "Impressions", "AVE"], axis=0,
                                            ascending=[True, True, False, False])
                    # Save duplicate URLS
                    dupe_urls = data[data['URL_Helper'].duplicated(keep='first') == True]

                    # Remove duplicate URLS
                    data = data[~data['URL_Helper'].duplicated(keep='first') == True]

                    # Drop URL Helper column from both dfs
                    data.drop(["URL_Helper"], axis=1, inplace=True, errors='ignore')
                    dupe_urls.drop(["URL_Helper"], axis=1, inplace=True, errors='ignore')

                    frames = [data, blank_urls]
                    data = pd.concat(frames)

                    ### Dupe column cleaning ###

                    # Split off records with blank headline/outlet/type
                    blank_set = data[data.Headline.isna() | data.Outlet.isna() | data.Type.isna()]
                    data = data[~data.Headline.isna() | data.Outlet.isna() | data.Type.isna()]

                    # Add helper column
                    data["dupe_helper"] = data['Type'].astype('string') + data['Outlet'].astype('string') + data['Headline']
                    data = data.sort_values(["dupe_helper", "Author", "Impressions", "AVE"], axis=0,
                                            ascending=[True, True, False, False])
                    dupe_cols = data[data['dupe_helper'].duplicated(keep='first') == True]
                    data = data[~data['dupe_helper'].duplicated(keep='first') == True]

                    # Drop helper column and rejoin broadcast
                    data.drop(["dupe_helper"], axis=1, inplace=True, errors='ignore')
                    dupe_cols.drop(["dupe_helper"], axis=1, inplace=True, errors='ignore')
                    frames = [data, broadcast, blank_set]
                    traditional = pd.concat(frames)
                    dupes = pd.concat([dupe_urls, dupe_cols])

                    with col3:
                        st.write('✓ Duplicates Removed')

                    if len(traditional) > 0:
                        with st.expander("Traditional"):
                            st.markdown('(First 50 rows)')
                            # st.dataframe(traditional.head(50).style.format(format_dict))
                            st.dataframe(traditional.head(50))
                            st.markdown('(First 50 rows)')
                            # st.dataframe(data.head(50).style.format(format_dict))
                    if len(social) > 0:
                        with st.expander("Social"):
                            st.markdown('(First 50 rows)')
                            # st.dataframe(social.head(50).style.format(format_dict))
                            st.dataframe(social.head(50))
                    if len(dupes) > 0:
                        with st.expander("Deleted Duplicates"):
                            # st.dataframe(dupes.style.format(format_dict))
                            st.dataframe(dupes)

                    original_trad_auths = top_x_by_mentions(traditional, "Author")
                    st.session_state.original_trad_auths = original_trad_auths
                    st.session_state.df_traditional = traditional
                    st.session_state.df_social = social
                    st.session_state.df_dupes = dupes
                    st.session_state.standard_step = True

elif page == "3: Impressions - Outliers":
    traditional = st.session_state.df_traditional
    st.title('Impressions - Outliers')
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')

    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    elif len(traditional) == 0:
        st.subheader("No traditional media in data. Skip to next step.")

    else:
        st.subheader('Check highest impressions numbers:')
        outliers = traditional[['Outlet', 'Type', 'Impressions', 'Headline', 'URL', 'Country']].nlargest(100,
                                                                                                         'Impressions')
        outliers.index.name = 'Row'
        st.dataframe(outliers.style.format(format_dict))
        outlier_index = outliers.index.values.tolist()

        with st.form("Update Outliers", clear_on_submit=True):
            st.subheader("Update Impressions Outliers")
            index_numbers = st.multiselect('Row index number(s): ', outlier_index,
                                           help='Select the row number from the table above.')
            new_impressions_value = int(st.number_input('New impressions value for row(s)', step=1,
                                                        help='Write in the new impression value for the selected row.'))
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

    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')

    elif len(traditional) == 0:
        st.warning("No traditional media in data. Skip to next step.")

    elif st.session_state.filled == True:
        st.success("Missing impressions fill complete!")



    elif st.session_state.outliers == False:
        st.warning('Please confirm outliers step is complete before running this step.')
        done_outliers = st.button('Done with outliers')
        if done_outliers:
            st.session_state.outliers = True
            st.experimental_rerun()

    else:
        blank_impressions = traditional['Impressions'].isna().sum()

        if blank_impressions == 0:
            st.info('No missing impressions numbers in data')

        else:

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



elif page == "5: Authors - Missing":
    st.title('Authors - Missing')
    traditional = st.session_state.df_traditional
    original_trad_auths = st.session_state.original_trad_auths

    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')

    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    elif len(traditional) == 0:
        st.subheader("No traditional media in data. Skip to next step.")

    else:
        counter = st.session_state.counter
        original_top_authors = st.session_state.original_auths


        def fix_author(df, headline_text, new_author):
            """Updates all authors for a given headline"""
            df.loc[df["Headline"] == headline_text, "Author"] = new_author


        def headline_authors(df, headline_text):
            """Returns the various authors for a given headline"""
            headline_authors = (df[df.Headline == headline_text].Author.value_counts().reset_index())
            return headline_authors


        # headline_table = pd.pivot_table(traditional, index="Headline", values=["Mentions", "Author"], aggfunc="count")
        # headline_table["Missing"] = headline_table["Mentions"] - headline_table["Author"]
        # headline_table = headline_table[headline_table["Author"] > 0]
        # headline_table = headline_table[headline_table['Missing'] > 0]
        # headline_table = headline_table.sort_values("Missing", ascending=False)
        # headline_table = headline_table.reset_index()
        # headline_table.rename(columns={'Author': 'Known',
        #                                'Mentions': 'Total'},
        #                       inplace=True, errors='raise')

        headline_table = data[['Headline', 'Mentions', 'Author']]
        headline_table = headline_table.groupby("Headline").count()
        headline_table["Missing"] = headline_table["Mentions"] - headline_table["Author"]
        headline_table = headline_table[(headline_table["Author"] > 0) & (headline_table['Missing'] > 0)].sort_values(
            "Missing", ascending=False)
        headline_table.rename(columns={'Author': 'Known', 'Mentions': 'Total'},
                              inplace=True, errors='raise')

        temp_headline_list = headline_table
        if counter < len(temp_headline_list):
            # headline_text = temp_headline_list.iloc[counter]['Headline']
            headline_text = temp_headline_list.index[counter]

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

            col1, col2, col3 = st.columns([12, 1, 9])
            with col1:
                st.subheader("Headline")
                st.table(headline_table.iloc[[counter]])
            with col2:
                st.write(" ")
            with col3:
                st.subheader("Authors in CSV")
                st.table(headline_authors(traditional, headline_text).rename(columns={'index': 'Possible Author(s)',
                                                                                  'Author': 'Matches'}))

            with st.form('auth updater', clear_on_submit=True):

                col1, col2, col3 = st.columns([8, 1, 8])
                with col1:
                    box_author = st.selectbox('Pick from possible Authors', possibles,
                                    help='Pick from one of the authors already associated with this headline.')

                with col2:
                    st.write(" ")
                    st.subheader("OR")

                with col3:
                    string_author = st.text_input("Write in the author name",
                                    help='Override above selection by writing in a custom name.')

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
            st.dataframe(original_trad_auths)

        with col2:
            st.subheader("New Top Authors")
            st.dataframe(top_x_by_mentions(traditional, "Author"))

        # st.subheader("Fixable Author Stats")
        # stats = (fixable_headline_stats(traditional, primary="Headline", secondary="Author"))
        # st.text(stats)


elif page == "6: Authors - Outlets":
    st.title("Author - Outlets")
    from unidecode import unidecode
    import requests
    from requests.structures import CaseInsensitiveDict

    traditional = st.session_state.df_traditional
    auth_outlet_skipped = st.session_state.auth_outlet_skipped
    auth_outlet_table = st.session_state.auth_outlet_table
    top_auths_by = st.session_state.top_auths_by

    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
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


        def reset_skips():
            st.session_state.auth_outlet_skipped = 0


        def name_match(series):
            non_match = 'color: #985331;'
            match = 'color: goldenrod'
            return [non_match if cell_value != author_name else match for cell_value in series]



        top_auths_by = st.selectbox('Top Authors by: ', ['Mentions', 'Impressions'], on_change=reset_skips)
        st.session_state.top_auths_by = top_auths_by
        if len(auth_outlet_table) == 0:
            if top_auths_by == 'Mentions':
                auth_outlet_table = traditional[['Author', 'Mentions', 'Impressions']].groupby(
                    by=['Author']).sum().sort_values(
                    ['Mentions', 'Impressions'], ascending=False).reset_index()
                auth_outlet_table['Outlet'] = ''
                auth_outlet_todo = auth_outlet_table

            if top_auths_by == 'Impressions':
                auth_outlet_table = traditional[['Author', 'Mentions', 'Impressions']].groupby(
                    by=['Author']).sum().sort_values(
                    ['Impressions', 'Mentions'], ascending=False).reset_index()
                auth_outlet_table['Outlet'] = ''
                auth_outlet_todo = auth_outlet_table

        else:
            if top_auths_by == 'Mentions':
                auth_outlet_table = auth_outlet_table.sort_values(['Mentions', 'Impressions'], ascending=False)#.reset_index()
                auth_outlet_todo = auth_outlet_table.loc[auth_outlet_table['Outlet'] == '']

            if top_auths_by == 'Impressions':
                auth_outlet_table = auth_outlet_table.sort_values(['Impressions', 'Mentions'], ascending=False)#.reset_index()
                auth_outlet_todo = auth_outlet_table.loc[auth_outlet_table['Outlet'] == '']


        auth_outlet_skipped = st.session_state.auth_outlet_skipped

        if auth_outlet_skipped < len(auth_outlet_todo):
            author_name = auth_outlet_todo.iloc[auth_outlet_skipped]['Author']

            # NAME, SKIP & RESET SKIP SECTION
            col1, but1, but2 = st.columns([2, 1, 1])
            with col1:
                st.markdown("""
                                <h2 style="color: goldenrod; padding-top:0!important; margin-top:-"> 
                                """ + author_name +
                            """</h2>
                            <style>.css-12w0qpk {padding-top:22px !important}</style>
                            """, unsafe_allow_html=True)
            with but1:
                next_auth = st.button('Skip to Next Author')
                if next_auth:
                    auth_outlet_skipped += 1
                    st.session_state.auth_outlet_skipped = auth_outlet_skipped
                    st.experimental_rerun()

                with but2:
                    reset_counter = st.button('Reset Skips')
                    if reset_counter:
                        auth_outlet_skipped = 0
                        st.session_state.auth_outlet_skipped = auth_outlet_skipped
                        st.experimental_rerun()

            search_results = fetch_outlet(unidecode(author_name))
            number_of_authors = len(search_results['results'])
            db_outlets = []

            if search_results['results'] == []:
                matched_authors = []
            elif search_results['results'] == None:
                matched_authors = []
            else:
                response_results = search_results['results']
                outlet_results = []

                for result in response_results:
                    auth_name = result['firstName'] + " " + result['lastName']
                    job_title = result['primaryEmployment']['jobTitle']
                    outlet = result['primaryEmployment']['outletName']
                    if result['country'] == None:
                        country = ''
                    # elif result['country'] == []:
                    #     country = ''
                    else:
                        country = result['country']['name']
                    auth_tuple = (auth_name, job_title, outlet, country)
                    outlet_results.append(auth_tuple)

                matched_authors = pd.DataFrame.from_records(outlet_results,
                                                            columns=['Name', 'Title', 'Outlet', 'Country'])
                matched_authors.loc[matched_authors.Outlet == "[Freelancer]", "Outlet"] = "Freelance"

                db_outlets = matched_authors.Outlet.tolist()

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
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([8, 1, 16])
            with col1:
                st.subheader("Outlets in CSV")  #########################################
                outlets_in_coverage = traditional.loc[traditional.Author == author_name].Outlet.value_counts()
                outlets_in_coverage_list = outlets_in_coverage.index
                outlets_in_coverage_list = outlets_in_coverage_list.insert(0, "Freelance")
                outlets_in_coverage = outlets_in_coverage.rename_axis('Outlet').reset_index(name='Matches')

                st.dataframe(outlets_in_coverage.style.apply(
                    lambda x: ['background: goldenrod; color: black' if v in db_outlets else "" for v in x],
                    axis=1))

            with col2:
                st.write(" ")

            with col3:
                st.subheader("Media Database Results")  #####################################
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
                        # country = result['country']['name']
                        if result['country'] == None:
                            country = 'None'
                        # elif result['country'] == []:
                        #     country = 'Blank'
                        else:
                            country = result['country']['name']
                        auth_tuple = (auth_name, job_title, outlet, country)
                        outlet_results.append(auth_tuple)

                    matched_authors = pd.DataFrame.from_records(outlet_results,
                                                                columns=['Name', 'Title', 'Outlet', 'Country'])
                    matched_authors.loc[matched_authors.Outlet == "[Freelancer]", "Outlet"] = "Freelance"




                    st.table(matched_authors.style.apply(lambda x: [
                        'background: goldenrod; color: black' if v in outlets_in_coverage.Outlet.tolist() else "" for v
                        in x], axis=1).apply(name_match, axis=0, subset='Name'))

                    possibles = matched_authors.Outlet

            # FORM TO UPDATE AUTHOR OUTLET ######################
            with st.form('auth updater', clear_on_submit=True):
                # if len(matched_authors) > 0:
                col1, col2, col3 = st.columns([8, 1, 8])
                with col1:
                    if len(matched_authors) > 0:
                        box_outlet = st.selectbox('Pick outlet from DATABASE MATCHES', possibles,
                                                  help='Pick from one of the outlets associated with this author name.')

                    else:
                        box_outlet = st.selectbox('Pick outlet from COVERAGE or "Freelance"', outlets_in_coverage_list)

                with col2:
                    st.write(" ")
                    st.subheader("OR")
                with col3:
                    string_outlet = st.text_input("Write in an outlet name",
                                              help='Override above selection by writing in a custom name.')

                submitted = st.form_submit_button("Assign Outlet")

            if submitted:
                if len(string_outlet) > 0:
                    new_outlet = string_outlet
                else:
                    new_outlet = box_outlet

                auth_outlet_table.loc[auth_outlet_table["Author"] == author_name, "Outlet"] = new_outlet
                # auth_outlet_skipped += 1
                st.session_state.auth_outlet_skipped = auth_outlet_skipped
                st.session_state.auth_outlet_table = auth_outlet_table
                st.experimental_rerun()

            col1, col2, col3 = st.columns([8, 1, 4])
            with col1:
                st.subheader("Top Authors")
                if 'Outlet' in auth_outlet_table.columns:
                    if top_auths_by == 'Mentions':
                        st.table(
                            auth_outlet_table[['Author', 'Outlet', 'Mentions', 'Impressions']].fillna('').sort_values(
                                ['Mentions', 'Impressions'], ascending=False).head(15).style.format(format_dict))
                    if top_auths_by == 'Impressions':
                        st.table(
                            auth_outlet_table[['Author', 'Outlet', 'Mentions', 'Impressions']].fillna('').sort_values(
                                ['Impressions', 'Mentions'], ascending=False).head(15).style.format(format_dict))

                else:
                    st.table(auth_outlet_table[['Author', 'Mentions', 'Impressions']].fillna('').head(15).style.format(
                        format_dict))
            with col2:
                st.write(" ")
            with col3:
                st.subheader('Outlets assigned')

                if 'Outlet' in auth_outlet_table.columns:
                    assigned = len(auth_outlet_table.loc[auth_outlet_table['Outlet'] != ''])
                    st.metric(label='Assigned', value=assigned)
                else:
                    st.metric(label='Assigned', value=0)
        else:
            st.write("You've reached the end of the list!")
            st.write(f"Counter: {auth_outlet_skipped}")
            st.write(f"To Do: {len(auth_outlet_todo)}")
            if auth_outlet_skipped > 0:
                reset_counter = st.button('Reset Counter')
                if reset_counter:
                    auth_outlet_skipped = 0
                    st.session_state.auth_outlet_skipped = auth_outlet_skipped
                    st.experimental_rerun()
            else:
                st.write("✓ Nothing left to update here.")


elif page == "7: Translation":
    st.title('Translation')
    from deep_translator import GoogleTranslator
    from concurrent.futures import ThreadPoolExecutor
    from titlecase import titlecase

    traditional = st.session_state.df_traditional
    social = st.session_state.df_social
    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    elif st.session_state.translated_headline == True and st.session_state.translated_snippet == True and st.session_state.translated_summary == True:
        st.subheader("✓ Translation complete.")
        trad_non_eng = len(traditional[traditional['Language'] != 'English'])
        soc_non_eng = len(social[social['Language'] != 'English'])

        if trad_non_eng > 0:
            with st.expander("Traditional - Non-English"):
                st.dataframe(traditional[traditional['Language'] != 'English'][
                                 ['Outlet', 'Headline', 'Snippet', 'Summary', 'Language', 'Country']])

        if soc_non_eng > 0:
            with st.expander("Social - Non-English"):
                st.dataframe(social[social['Language'] != 'English'][
                                 ['Outlet', 'Snippet', 'Summary', 'Language', 'Country']])
    elif len(traditional[traditional['Language'] != 'English']) == 0 and len(
            social[social['Language'] != 'English']) == 0:
        st.subheader("No translation required")
    else:
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
            non_english_records = len(traditional[traditional['Language'] != 'English']) + len(
                social[social['Language'] != 'English'])
            minutes = non_english_records // 100
            if minutes == 0:
                min_word = 'minute'
            else:
                min_word = 'minutes'
            st.write(f"There are {non_english_records} non-English records in your data.")


        translation_stats_combo()
        if len(traditional) > 0:
            with st.expander("Traditional - Non-English"):
                st.dataframe(traditional[traditional['Language'] != 'English'][
                                 ['Outlet', 'Headline', 'Snippet', 'Summary', 'Language', 'Country']])

        if len(social) > 0:
            with st.expander("Social - Non-English"):
                st.dataframe(social[social['Language'] != 'English'][
                                 ['Outlet', 'Snippet', 'Summary', 'Language', 'Country']])

        with st.form('translation_form'):
            st.subheader("Pick columns for translations")
            st.warning("WARNING: Translation will over-write the original text.")

            if len(traditional) > 0:
                if st.session_state.translated_headline == False:
                    headline_to_english = st.checkbox('Headline', value=True)
                else:
                    st.success('✓ Headlines translated.')
                    headline_to_english = False
            else:
                headline_to_english = False

            if st.session_state.translated_snippet == False:
                snippet_to_english = st.checkbox('Snippet', value=True)
            else:
                st.success('✓ Snippets translated.')
                snippet_to_english = False

            if st.session_state.translated_summary == False:
                summary_to_english = st.checkbox('Summary', value=True)
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


elif page == "8: Review":
    st.title('Review')
    # import altair as alt

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

                trend = traditional.groupby('Date')[['Mentions']].sum()
                trend.plot()

                # trend = alt.Chart(traditional).mark_line().encode(
                #     x='Date:T',
                #     y='count(Mentions):Q'
                # )
                # st.altair_chart(trend, use_container_width=True)
                #
                # st.markdown('##')
                # st.subheader('Impressions Trend')
                #
                # trend2 = alt.Chart(traditional).mark_line().encode(
                #     x='Date:T',
                #     y='sum(Impressions):Q'
                # )
                # st.altair_chart(trend2, use_container_width=True)

                # st.subheader("Cleaned Data")
                # st.dataframe(traditional.style.format(format_dict))
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

                # st.markdown('##')
                # st.subheader('Mention Trend')
                #
                # trend = alt.Chart(social).mark_line().encode(
                #     x='Date:T',
                #     y='count(Mentions):Q'
                # )
                # st.altair_chart(trend, use_container_width=True)
                #
                # st.markdown('##')
                # st.subheader('Impressions Trend')
                #
                # trend2 = alt.Chart(social).mark_line().encode(
                #     x='Date:T',
                #     y='sum(Impressions):Q'
                # )
                # st.altair_chart(trend2, use_container_width=True)

                # st.subheader("Cleaned Data")
                # st.dataframe(social.style.format(format_dict))
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


elif page == "9: Download":
    st.title('Download')
    import io

    if st.session_state.upload_step == False:
        st.error('Please upload a CSV before trying this step.')
    elif st.session_state.standard_step == False:
        st.error('Please run the Standard Cleaning before trying this step.')
    else:
        export_name = st.session_state.export_name

        traditional = st.session_state.df_traditional
        social = st.session_state.df_social
        dupes = st.session_state.df_dupes
        uncleaned = st.session_state.df_uncleaned
        auth_outlet_table = st.session_state.auth_outlet_table

        traditional['Date'] = pd.to_datetime(traditional['Date'])
        social['Date'] = pd.to_datetime(social['Date'])
        dupes['Date'] = pd.to_datetime(dupes['Date'])

        with st.form("my_form_download"):
            st.subheader("Generate your cleaned data workbook")
            submitted = st.form_submit_button("Go!")
            if submitted:
                with st.spinner('Building workbook now...'):

                    output = io.BytesIO()
                    writer = pd.ExcelWriter(output, engine='xlsxwriter', datetime_format='yyyy-mm-dd',
                                            options={'in_memory': True})
                    workbook = writer.book
                    cleaned_dfs = []
                    cleaned_sheets = []

                    if len(traditional) > 0:
                        traditional = traditional.sort_values(by=['Impressions'], ascending=False)
                        traditional.to_excel(writer, sheet_name='CLEAN TRAD', startrow=1, header=False, index=False)
                        worksheet1 = writer.sheets['CLEAN TRAD']
                        worksheet1.set_tab_color('black')
                        cleaned_dfs.append((traditional, worksheet1))
                        cleaned_sheets.append(worksheet1)

                    if len(social) > 0:
                        social = social.sort_values(by=['Impressions'], ascending=False)
                        social.to_excel(writer, sheet_name='CLEAN SOCIAL', startrow=1, header=False, index=False)
                        worksheet2 = writer.sheets['CLEAN SOCIAL']
                        worksheet2.set_tab_color('black')
                        cleaned_dfs.append((social, worksheet2))
                        cleaned_sheets.append(worksheet2)

                    if len(auth_outlet_table) > 0:
                        authors = auth_outlet_table.sort_values(by=['Mentions', 'Impressions'], ascending=False)
                        authors.to_excel(writer, sheet_name='Authors', header=True, index=False)
                        worksheet5 = writer.sheets['Authors']
                        worksheet5.set_tab_color('green')

                    if len(dupes) > 0:
                        dupes.to_excel(writer, sheet_name='DLTD DUPES', header=True, index=False)
                        worksheet3 = writer.sheets['DLTD DUPES']
                        worksheet3.set_tab_color('#c26f4f')
                        cleaned_dfs.append((dupes, worksheet3))
                        cleaned_sheets.append(worksheet3)

                    uncleaned.to_excel(writer, sheet_name='RAW', header=True, index=False)
                    worksheet4 = writer.sheets['RAW']
                    worksheet4.set_tab_color('#c26f4f')

                    for clean_df in cleaned_dfs:
                        (max_row, max_col) = clean_df[0].shape
                        column_settings = [{'header': column} for column in clean_df[0].columns]
                        clean_df[1].add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

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
