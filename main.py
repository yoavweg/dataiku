import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import xgboost as xgb

columns = ['age', 'class of worker', 'detailed industry code', 'detailed occupation code', 'education',
           'wage per hour', 'enroll in edu inst last wk', 'marital stat', 'major industry code',
           'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union',
           'reason for unemployment', 'full or part time employment stat', 'capital gains', 'capital losses',
           'dividends from stocks', 'tax filer stat', 'region of previous residence', 'state of previous residence',
           'detailed household and family stat', 'detailed household summary in household', 'instance weight',
           'migration code-change in msa', 'migration code-change in reg', 'migration code-move within reg',
           'live in this house 1 year ago', 'migration prev res in sunbelt', 'num persons worked for employer',
           'family members under 18', 'country of birth father', 'country of birth mother', 'country of birth self',
           'citizenship', 'own business or self employed', "fill inc questionnaire for veteran's admin",
           'veterans benefits', 'weeks worked in year', 'year', 'target']
mapped_columns = {'age': 'AAGE',
                  'class of worker': 'ACLSWKR',
                  'detailed industry code': 'ADTIND',
                  'detailed occupation code': 'ADTOCC',
                  'adjusted gross income': 'AGI',
                  'education': 'AHGA',
                  'wage per hour': 'AHRSPAY',
                  'enroll in edu inst last wk': 'AHSCOL',
                  'marital stat': 'AMARITL',
                  'major industry code': 'AMJIND',
                  'major occupation code': 'AMJOCC',
                  'race': 'ARACE',
                  'hispanic origin': 'AREORGN',
                  'sex': 'ASEX',
                  'member of a labor union': 'AUNMEM',
                  'reason for unemployment': 'AUNTYPE',
                  'full or part time employment stat': 'AWKSTAT',
                  'capital gains': 'CAPGAIN',
                  'capital losses': 'CAPLOSS',
                  'dividends from stocks': 'DIVVAL',
                  'tax filer stat': 'FILESTAT',
                  'region of previous residence': 'GRINREG',
                  'state of previous residence': 'GRINST',
                  'detailed household and family stat': 'HHDFMX',
                  'detailed household summary in household': 'HHDREL',
                  'instance weight': 'MARSUPWT',
                  'migration code-change in msa': 'MIGMTR1',
                  'migration code-change in reg': 'MIGMTR3',
                  'migration code-move within reg': 'MIGMTR4',
                  'live in this house 1 year ago': 'MIGSAME',
                  'migration prev res in sunbelt': 'MIGSUN',
                  'num persons worked for employer': 'NOEMP',
                  'family members under 18': 'PARENT',
                  'total person earnings': 'PEARNVAL',
                  'country of birth father': 'PEFNTVTY',
                  'country of birth mother': 'PEMNTVTY',
                  'country of birth self': 'PENATVTY',
                  'citizenship': 'PRCITSHP',
                  'total person income': 'PTOTVAL',
                  'own business or self employed': 'SEOTR',
                  'taxable income amount': 'TAXINC',
                  "fill inc questionnaire for veteran's admin": 'VETQVA',
                  'veterans benefits': 'VETYN',
                  'weeks worked in year': 'WKSWORK',
                  'year': 'YEAR',
                  'target': 'income'}


def preprocess(df):
    df.columns = columns
    df.columns = df.columns.map(mapped_columns)
    df = df.replace(to_replace=' Not in universe', value=np.nan)
    df = df.replace(' Not in universe under 1 year old', np.nan)
    df = df.drop_duplicates()
    df['income'] = df['income'].apply(lambda x: 1 if x == ' 50000+.' else 0)  # Encode target variable as binary
    df = df[~df.ACLSWKR.isin([' Never worked', ' Without pay'])]
    df['CLSWKR'] = df.ACLSWKR.map({' Self-employed-not incorporated': 'self',
                                   ' Self-employed-incorporated': 'self',
                                   ' State government': 'gov',
                                   ' Federal government': 'gov',
                                   ' Local government': 'gov',
                                   ' Private': 'private'})
    df['is_year_94'] = df.YEAR.map({94: 1, 95: 0})
    df = df[df.AHGA.isin([" Prof school degree (MD DDS DVM LLB JD)", " Doctorate degree(PhD EdD)",
                          " Masters degree(MA MS MEng MEd MSW MBA)", " Bachelors degree(BA AB BS)",
                          " Associates degree-academic program", " Associates degree-occup /vocational",
                          " Some college but no degree", " High school graduate"])]
    df['HSCOL'] = df.AHSCOL.map({' High school': 1, ' College or university': 2})
    df.HSCOL.fillna(0, inplace=True)
    df['MARITL'] = df.AMARITL.map({' Never married': 0,
                                   ' Married-civilian spouse present': 1, ' Married-A F spouse present': 1,
                                   ' Divorced': 2, ' Widowed': 2, ' Separated': 2, ' Married-spouse absent': 2,
                                   })
    df['is_male'] = df.ASEX.map({' Male': 1, ' Female': 0})
    df['is_union'] = df.AUNMEM.map({' No': 0, ' Yes': 1})
    df['employment_type'] = df.AWKSTAT.map({' Children or Armed Forces': 2, ' Full-time schedules': 2,
                                            'Not in labor force': 0, ' PT for non-econ reasons usually FT': 1,
                                            ' Unemployed full-time': 0, ' PT for econ reasons usually PT': 1,
                                            ' Unemployed part- time': 0, ' PT for econ reasons usually FT': 1})
    df['capital_change'] = df.CAPGAIN - df.CAPLOSS + df.DIVVAL
    df['tax_file_type'] = df.FILESTAT.map({' Nonfiler': 0, ' Joint both under 65': 2, ' Single': 1,
                                           ' Joint both 65+': 2, ' Head of household': 1,
                                           ' Joint one under 65 & one 65+': 2})
    df['household_stat'] = df.HHDREL.map({' Householder': 3, ' Child under 18 never married': 0,
                                          ' Spouse of householder': 2, ' Child 18 or older': 0,
                                          ' Other relative of householder': 0, ' Nonrelative of householder': 1,
                                          ' Group Quarters- Secondary individual': 0,
                                          ' Child under 18 ever married': 0})
    df['same_house'] = df.MIGSAME.map({' Yes': 1, ' No': 0})
    df['is_native'] = df.PRCITSHP.str.contains('Native')
    df['AMJOCC'] = df['AMJOCC'].fillna('Other service')
    df['CLSWKR'] = df['CLSWKR'].fillna('Unknown')
    # y[x_train.employment_type.isna()].value_counts()
    # income
    # 0    15324
    # 1    437
    df['employment_type'] = df['employment_type'].fillna(0)

    df = df.drop(columns=['ACLSWKR', 'AMARITL', 'ADTIND', 'ADTOCC', 'AREORGN', 'AWKSTAT', 'AUNTYPE', 'GRINREG',
                          'HHDFMX', 'MIGMTR1', 'MIGMTR3',
                          'MIGMTR4', 'MIGSUN', 'VETQVA', 'ASEX', 'AUNMEM', 'AUNMEM', 'CAPLOSS', 'CAPGAIN',
                          'DIVVAL', 'FILESTAT', 'HHDREL', 'MIGSAME', 'PRCITSHP', 'YEAR', 'GRINST',
                          'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'AHSCOL', 'PARENT'])
    return df


def main():
    train = pd.read_csv('data/census_income_learn.csv')
    test = pd.read_csv('data/census_income_test.csv')

    # Identify categorical columns that need one-hot encoding
    categorical_cols = ['AHGA', 'AMJIND', 'AMJOCC', 'ARACE', 'employment_type', 'SEOTR', 'VETYN', 'MARITL',
                        'HSCOL', 'tax_file_type', 'household_stat', 'CLSWKR']

    # Identify continuous columns that need scaling
    continuous_cols = ['AAGE', 'AHRSPAY', 'WKSWORK', 'capital_change', 'NOEMP']

    x_train = preprocess(train)
    x_test = preprocess(test)

    w = train['MARSUPWT'].copy()  # weights
    c = x_train.apply(lambda x: pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
    print(c)
    g = sns.barplot(x='income', y='AHGA', data=x_train, orient='h')
    g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=8)
    g.set_yticklabels(g.get_yticklabels(), rotation=45, fontsize=8)
    g.set_ylabel('education')
    g.set_title('Income level by education level')

    y = x_train['income'].copy()
    y_test = x_test['income'].copy()
    x_train.drop(columns=['income', 'MARSUPWT'], inplace=True)
    x_test.drop(columns=['income', 'MARSUPWT'], inplace=True)
    # Create transformers for preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols),
            ('cont', StandardScaler(), continuous_cols)
        ])

    # Apply transformations
    train_trans = preprocessor.fit_transform(x_train)
    test_trans = preprocessor.fit_transform(x_test)
    model = xgb.XGBClassifier()
    model.fit(train_trans, y)
    y_pred = model.predict(test_trans)
    print(
        f"Recall: {recall_score(y_test, y_pred):0.2f}\nPrecision: {precision_score(y_test, y_pred):0.2f}\nF1: {f1_score(y_test, y_pred):0.2f}")
    print(confusion_matrix(y_test, y_pred, normalize='all'))
    xgb.plot_importance(model)
    # resample
    sm = SMOTE(random_state=0)
    x_res, y_res = sm.fit_resample(train_trans, y)
    model_res = xgb.XGBClassifier()
    model_res.fit(x_res, y_res)
    y_pred_res = model_res.predict(test_trans)
    print(
        f"Recall: {recall_score(y_test, y_pred_res):0.2f}\nPrecision: {precision_score(y_test, y_pred_res):0.2f}\nF1: {f1_score(y_test, y_pred_res):0.2f}")
    print(confusion_matrix(y_test, y_pred, normalize='all'))
    xgb.plot_importance(model_res)
    # categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', StandardScaler(), continuous_cols)
        ])
    x_cat = x_train.copy()
    x_cat[categorical_cols] = x_cat[categorical_cols].astype("category")
    x_cat[continuous_cols] = preprocessor.fit_transform(x_cat)
    model_cat = xgb.XGBClassifier(enable_categorical=True)
    model_cat.fit(x_cat, y)
    x_cat_test = x_test.copy()
    x_cat_test[categorical_cols] = x_cat_test[categorical_cols].astype("category")
    x_cat_test[continuous_cols] = preprocessor.fit_transform(x_cat_test)
    cat_pred = model_cat.predict(x_cat_test)
    print(
        f"Recall: {recall_score(y_test, cat_pred):0.2f}\nPrecision: {precision_score(y_test, cat_pred):0.2f}\nF1: {f1_score(y_test, cat_pred):0.2f}")
    print(confusion_matrix(y_test, y_pred, normalize='all'))
    xgb.plot_importance(model_cat)


if __name__ == "__main__":
    main()
