import pickle
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import tree
from xgboost import XGBClassifier
from DataManipulation import makeDatasetUniform
from DataManipulation.changeData import calculateRefScore, updateRefScore, createTeamNames, updateTeamOveralls, \
    aggPoints, get_gss, addTeamForm, get_form_points, get_3game_ws, get_5game_ws, get_3game_ls, get_5game_ls, set_mw


def featureSelectionTest(X_, Y_):
    fScore = f_classif(X_, Y_)
    fMutualScore = mutual_info_classif(X_, Y_)
    top10_best = SelectKBest(f_classif, k=10).fit_transform(X_, Y_)
    return fScore, fMutualScore, top10_best


def train(X, Y):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    # test_size=0.3, random_state=2)
    #scaler = StandardScaler()
    log = LogisticRegression()
    svc = SVC(random_state=912, kernel='rbf')
    dt = tree.DecisionTreeClassifier(random_state=42, max_depth=2, min_samples_leaf=4, min_samples_split=4)
    xgb = XGBClassifier(num_class=3,
                        n_estimators=40,
                        learning_rate=1e-4,
                        max_depth=2,
                        min_child_weight=3,
                        gamma=0.01,
                        colsample_bytree=0.5,
                        subsample=0.3,
                        )
    scaler = StandardScaler()
    scaler.fit_transform(X)

    log.fit(X, Y)
    svc.fit(X, Y)
    dt.fit(X, Y)
    xgb.fit(X, Y)

    log_fn = 'LogisticRegression.sav'
    svm_fn = 'SVC.sav'
    dt_fn = 'DT.sav'
    xgb_fn = 'XGBoost.sav'

    # log = train(LogisticRegression, X, Y)
    # svc = train(SVC, X, Y)
    # xgb_res = final_pipeline(xgbRes, test=False)

    Save_LogisticModel = pickle.dump(log, open(log_fn, 'wb'))
    Save_SvmModel = pickle.dump(svc, open(svm_fn, 'wb'))
    Save_DtModel = pickle.dump(dt, open(dt_fn, 'wb'))
    Save_XgbModel = pickle.dump(xgb, open(xgb_fn, 'wb'))


def getFeaturesOutput(ref, season, team, no, encode_ht, encode_at, encode_htr, encode_out):
    season = makeDatasetUniform.makeSeasonDataEven(season)
    team = makeDatasetUniform.makeTeamsDataEven(team)
    # Update ref_20 scores
    ref = calculateRefScore(ref)
    season = updateRefScore(season, ref)
    # Update Team scores
    team_overall = createTeamNames(team)
    season = updateTeamOveralls(season, team_overall)
    # Update team_20 points
    season = aggPoints(season)
    # Get gss
    season = get_gss(season)
    # Update Team Form
    season = addTeamForm(season, 5)
    season = addTeamForm(season, 4)
    season = addTeamForm(season, 3)
    season = addTeamForm(season, 2)
    season = addTeamForm(season, 1)
    season['HTFormPtsStr'] = season['HM1'] + season['HM2'] + season['HM3'] + season['HM4'] + season['HM5']
    season['ATFormPtsStr'] = season['AM1'] + season['AM2'] + season['AM3'] + season['AM4'] + season['AM5']
    season['HTFormPts'] = season['HTFormPtsStr'].apply(get_form_points)
    season['ATFormPts'] = season['ATFormPtsStr'].apply(get_form_points)
    season['HTWinStreak3'] = season['HTFormPtsStr'].apply(get_3game_ws)
    season['HTWinStreak5'] = season['HTFormPtsStr'].apply(get_5game_ws)
    season['HTLossStreak3'] = season['HTFormPtsStr'].apply(get_3game_ls)
    season['HTLossStreak5'] = season['HTFormPtsStr'].apply(get_5game_ls)
    season['ATWinStreak3'] = season['ATFormPtsStr'].apply(get_3game_ws)
    season['ATWinStreak5'] = season['ATFormPtsStr'].apply(get_5game_ws)
    season['ATLossStreak3'] = season['ATFormPtsStr'].apply(get_3game_ls)
    season['ATLossStreak5'] = season['ATFormPtsStr'].apply(get_5game_ls)
    # Set MW
    season = set_mw(season)

    # Get Goal Difference
    season['HTGD'] = season['HTGS'] - season['HTGC']
    season['ATGD'] = season['ATGS'] - season['ATGC']

    # Diff in points
    season['DiffPts'] = season['HTP'] - season['ATP']
    season['DiffFormPts'] = season['HTFormPts'] - season['ATFormPts']
    # Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
    cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
    season.MW = season.MW.astype(float)
    for col in cols:
        season[col] = season[col] / season.MW

    features, output = reformat_data(season, no, encode_ht, encode_at, encode_htr)
    output = encode_out.fit_transform(output)
    return features, output


def feature_preprocessing(X_features, no, encode_ht, encode_at, encode_htr):
    drop_cols = ['HM5', 'AM5', 'HM4', 'AM4', 'HM3', 'AM3', 'HM2', 'AM2', 'HM1', 'AM1', 'HTFormPtsStr', 'ATFormPtsStr']
    X_features = X_features.drop(columns=drop_cols, axis=1)
    X_features = X_features.drop(columns='Div', axis=1)
    X_features = X_features.drop(columns='Date', axis=1)
    if no == 20:
        X_features = X_features.drop(columns='Time', axis=1)
    X_features['HomeTeam'] = encode_ht.fit_transform(X_features['HomeTeam'])
    X_features['AwayTeam'] = encode_at.fit_transform(X_features['AwayTeam'])
    X_features['HTR'] = encode_htr.fit_transform(X_features['HTR'])
    return X_features


def reformat_data(df, no, encode_ht, encode_at, encode_htr):
    X_ = df.drop(columns='FTR', axis=1)
    Y_ = df['FTR']
    X_ = feature_preprocessing(X_, no, encode_ht, encode_at, encode_htr)
    return X_, Y_

# def final_pipeline(classifier, test):
#    pipe = Pipeline([('scaler', StandardScaler()), ('class', classifier)])
#    if test:
#        return pipe.score(X_true, Y_true)
#    else:
#        pipe.fit(X, Y)
#        return classifier


# Debug:
