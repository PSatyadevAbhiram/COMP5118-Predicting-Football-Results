import pandas as pd


def set_mw(df):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1) % 10) == 0:
            j = j + 1
    df['MW'] = MatchWeek
    return df


def get_goals_conceded(df):
    # Create a dictionary with team_20 names as keys
    teams = {}
    for i in df.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match location.
    for i in range(len(df)):
        ATGC = df.iloc[i]['FTHG']
        HTGC = df.iloc[i]['FTAG']
        teams[df.iloc[i].HomeTeam].append(HTGC)
        teams[df.iloc[i].AwayTeam].append(ATGC)

    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsConceded[0] = 0
    # Aggregate to get uptil that point
    for i in range(2, 39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i - 1]
    return GoalsConceded


def get_goals_scored(df):
    # Create a dictionary with team_20 names as keys
    teams = {}
    for i in df.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match location.
    for i in range(len(df)):
        HTGS = df.iloc[i]['FTHG']
        ATGS = df.iloc[i]['FTAG']
        teams[df.iloc[i].HomeTeam].append(HTGS)
        teams[df.iloc[i].AwayTeam].append(ATGS)

    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsScored[0] = 0
    # Aggregate to get uptil that point
    for i in range(2, 39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i - 1]
    return GoalsScored


def get_gss(df):
    GC = get_goals_conceded(df)
    GS = get_goals_scored(df)

    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = df.iloc[i].HomeTeam
        at = df.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1

    df['HTGS'] = HTGS
    df['ATGS'] = ATGS
    df['HTGC'] = HTGC
    df['ATGC'] = ATGC

    return df


# Identify Win/Loss Streaks if any.
def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0


def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0


def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0


def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0


# Gets the form points.
def get_form_points(string):
    sum_form_points = 0
    for letter in string:
        sum_form_points += points(letter)
    return sum_form_points


def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0


def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0


def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0


def points(res):
    if res == 'W':
        return 3
    elif res == 'D':
        return 1
    else:
        return 0


def total_points(res):
    matchres_points = res.applymap(points)
    for i in range(2, 39):
        matchres_points[i] = matchres_points[i] + matchres_points[i - 1]

    matchres_points.insert(column=0, loc=0, value=[0 * i for i in range(20)])
    return matchres_points


def aggPoints(df):
    matchres = matchResult(df)
    cum_pts = total_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = df.iloc[i].HomeTeam
        at = df.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1

    df['HTP'] = HTP
    df['ATP'] = ATP
    return df


def matchResult(df):
    teams = {}
    for i in df.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    for i in range(len(df)):
        if df.iloc[i].FTR == 'H':
            teams[df.iloc[i].HomeTeam].append('W')
            teams[df.iloc[i].AwayTeam].append('L')
        elif df.iloc[i].FTR == 'A':
            teams[df.iloc[i].HomeTeam].append('L')
            teams[df.iloc[i].AwayTeam].append('H')
        else:
            teams[df.iloc[i].HomeTeam].append('D')
            teams[df.iloc[i].AwayTeam].append('D')
    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T


def getForm(df, num):
    form = matchResult(df)
    form_final = form.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i - j]
            j += 1
    return form_final


def addTeamForm(df, num):
    form = getForm(df, num)
    h = ['M' for _ in range(num * 10)]
    a = ['M' for _ in range(num * 10)]

    j = num
    for i in range((num * 10), 380):
        ht = df.iloc[i].HomeTeam
        at = df.iloc[i].AwayTeam

        past = form.loc[ht][j]
        h.append(past[num - 1])

        past = form.loc[at][j]
        a.append(past[num - 1])

        if ((i + 1) % 10) == 0:
            j += 1
    df['HM' + str(num)] = h
    df['AM' + str(num)] = a

    return df


def updateTeamOveralls(s, t_overalls):
    home_team_names = s["HomeTeam"].tolist()
    away_team_names = s["AwayTeam"].tolist()
    home_team_ovr = []
    away_team_ovr = []
    tr = 0
    for name in home_team_names:
        arr = name.split(' ')
        i = 0
        conv_name = ""
        while i < len(arr):
            conv_name += arr[i][:3]
            i += 1
        # s.loc[s["HomeTeam"] == home_team_names[tr]] = conv_name
        s['HomeTeam'] = s['HomeTeam'].replace(name, conv_name)
        home_team_ovr.append(t_overalls[conv_name])
        tr += 1
    tr = 0
    for name in away_team_names:
        arr = name.split(' ')
        i = 0
        conv_name = ""
        while i < len(arr):
            conv_name += arr[i][:3]
            i += 1
        # s.loc[s["AwayTeam"] == away_team_names[tr]] = conv_name
        s['AwayTeam'] = s['AwayTeam'].replace(name, conv_name)
        away_team_ovr.append(t_overalls[conv_name])
        tr += 1
    s["HomeTeamOVR"] = home_team_ovr
    s["AwayTeamOVR"] = away_team_ovr
    return s


def createTeamNames(t):
    team_names = []
    ovr = []
    for col in t:
        if col == "Name":
            team_names = t[col].values.tolist()
        if col == "OVR":
            ovr = t[col].values.tolist()
    team_ovr = {}
    index = 0
    for name in team_names:
        arr = name.split(' ')
        i = 0
        conv_name = ""
        while i < len(arr):
            conv_name += arr[i][:3]
            i += 1
        team_ovr[conv_name] = ovr[index]
        index += 1
    return team_ovr


def updateRefScore(ds, rs):
    ref_names = []
    ref_index_count = 0
    for col in ds:
        if col == "Referee":
            ref_names = ds[col].values.tolist()
            break
        ref_index_count += 1
    for full_name in ref_names:
        name = full_name.split(' ')
        last_name = name[len(name) - 1]
        if last_name not in rs:
            rs[last_name] = sum(rs.values()) / len(rs)
    for full_name in ref_names:
        name = full_name.split(' ')
        last_name = name[len(name) - 1]
        ds['Referee'] = ds['Referee'].replace(full_name, rs[last_name])
    return ds


def calculateRefScore(df_ref):
    ref_score_dict = {}
    df_ref.sort_values('Referee')
    ref_names = []
    matches = []
    yellowCards = []
    redCards = []
    pens = []
    for col in df_ref:
        if col == "Referee":
            ref_names = df_ref[col].values.tolist()
        if col == "Matches":
            matches = df_ref[col].values.tolist()
        if col == "Red cards":
            redCards = df_ref[col].values.tolist()
        if col == "Yellow cards":
            yellowCards = df_ref[col].values.tolist()
    index = 0
    while index < len(matches):
        refName = ref_names[index].split(' ')
        ref_score_dict[refName[len(refName) - 1]] \
            = (yellowCards[index] + redCards[index]) / matches[index]
        index += 1
    return ref_score_dict

