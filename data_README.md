# User

user.csv

| UserID      | age    | gender | education | marriageStatus | haveBaby | hometown | residence |
| ----------- | ------ | ------ | --------- | -------------- | -------- | :------: | --------- |
| [1,2805118] | [0,80] | 1 2 0  | [0,7]     | [0,3]          | [0,6]    |   365种   | 400种      |

user_installedapps.csv

| UserID | appID |
| ------ | ----- |
|        |       |

user_app_actions.csv

| userID | installTime | appID |
| ------ | ----------- | ----- |
|        |             |       |

# AD

ad.csv

| creativeID | adID     | camgaignID | advertiserID | appID       | appPlatform |
| ---------- | -------- | ---------- | ------------ | ----------- | ----------- |
| [1,6582]   | [1,3616] | [1,720]    | [1,91]       | [14,472]非连续 | 0 1 2       |

app_categories.csv

| appID                  | APPCategory    |
| ---------------------- | -------------- |
| [14,433269]非连续 217041种 | 28种 [0,503]非连续 |

# Background

position.csv

| positionID | sitesetID | positionType |
| ---------- | --------- | ------------ |
| [1,7645]   | 0 1 2     | [0,5]        |

# Train

shape (3749528, 8)

| label | clickTime            | conversionTime | creativeID | userID   | positionID | connectionType | telecomsOperator |
| ----- | -------------------- | -------------- | ---------- | -------- | ---------- | -------------- | ---------------- |
|       | 170000到302359共20160种 |                |            | 2595627种 | 7219种      | [0,4]          | [0,3]            |

# Test

| instanceID | label | clickTime     | creativeID | userID | positionID | connectionType | telecomsOperator |
| ---------- | ----- | ------------- | ---------- | ------ | ---------- | -------------- | ---------------- |
|            | -1    | 310000到312359 |            |        |            | [0,4]          | [0,3]            |