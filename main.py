# 1- Which type of data can be used while creating a series object in pandas?
# Answer: While creating a series object in pandas we can use any type of data

import pandas as pd

month_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
data = pd.Series(month_nums, index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept','Oct', 'Nov', 'Dec'])
print(data)

students_num = {
    'MATDAIS': 29,
    'MATMIE': 30,
    'COMIE': 20,
    'COMEC': 30,
    }

data = pd.Series(students_num)
print(data)

exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, True, 9, 20, 14.5, 'abc', 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes'],
    }
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
new_data = pd.DataFrame(exam_data, index=labels)
print(new_data)


res = new_data[new_data['attempts'] > 2]
print(res)