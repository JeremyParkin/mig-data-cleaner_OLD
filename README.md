# mig-data-cleaner

############
QUICK START
############

1. Only works with CSV files exported from Agility Saved Coverage Folders
2. After uploading, the next step, "Standard Cleaning" is the only required step - all the rest are optional, but very useful


############
STEP DETAILS
############

1: Getting Started
  - upload your file and see basic stats on the uncleaned data

2: Standard Cleaning
  - Columns Renamed
  - Junk Columns Dropped
  - "(Online)" Removed from outlet names
  - AP Style Capitalization of Headlines
  - Media Types Cleaned
  - Columns Sorted
  - Tags Expanded
  - Yahoo Standardization
  - Original URLs Merged
  - Extra Spaces Removed
  - Social Split Out
  - Duplicates Removed

3: Impressions - Outliers
  - Manual step to check highest impressions numbers and update as necessary

4: Impressions - Fill Blanks
  - Select a statistical fill level to apply to records with missing impressions numbers


5: Authors - Missing
  - Assign authors to matched headlines where some instances have author info and others do not  

6: Authors - Outlets
  - Match outlets to top authors using data from your CSV and from Agility's Media Database

7: Translation
  - Translate headlines, summaries and snippets to English

8: Review
  - check totals and other details for traditional, social and deleted duplicates

9: Download
  - Generate and download an xlsx file with all the cleaned data
