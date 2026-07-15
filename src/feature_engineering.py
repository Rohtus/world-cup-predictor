
import pandas as pd


def get_team_vector(
    team_name,
    team_vectors,
    statistical_columns,
    team_id_column="country_clean",
):
    """
    Retrieve the statistical vector of a national team.

    Parameters
    ----------
    team_name : str
        Name of the national team.

    team_vectors : pandas.DataFrame
        Dataset containing one aggregated vector per team.

    statistical_columns : list of str
        Statistical feature columns to retrieve.

    team_id_column : str, default="country_clean"
        Column used to identify each team.

    Returns
    -------
    pandas.Series
        Statistical vector of the requested team.

    Raises
    ------
    ValueError
        If the team is not available or appears more than once.
    """

    team_rows = team_vectors.loc[
        team_vectors[team_id_column].eq(team_name)
    ]

    if team_rows.empty:
        available_teams = sorted(
            team_vectors[team_id_column].unique()
        )

        raise ValueError(
            f"Team '{team_name}' was not found. "
            f"Available teams: {available_teams}"
        )

    if len(team_rows) != 1:
        raise ValueError(
            f"Expected one vector for '{team_name}', "
            f"but found {len(team_rows)}."
        )

    return (
        team_rows
        .iloc[0]
        .loc[statistical_columns]
        .copy()
    )

def build_current_match_vector(
    home_team,
    away_team,
    team_vectors,
    statistical_columns,
    team_id_column="country_clean",
):
    """
    Build a base current match vector from two national teams.

    Parameters
    ----------
    home_team : str
        Name of the home national team.

    away_team : str
        Name of the away national team.

    team_vectors : pandas.DataFrame
        Dataset containing one aggregated vector per team.

    statistical_columns : list of str
        Statistical feature columns used to represent each team.

    team_id_column : str, default="country_clean"
        Column used to identify each team.

    Returns
    -------
    pandas.DataFrame
        Single-row match vector containing home and away features.

    Raises
    ------
    ValueError
        If the same team is used as both home and away.
    """

    if home_team == away_team:
        raise ValueError(
            "Home team and away team must be different."
        )

    home_vector = get_team_vector(
    team_name=home_team,
    team_vectors=team_vectors,
    statistical_columns=statistical_columns,
    team_id_column=team_id_column,
    )

    away_vector = get_team_vector(
    team_name=away_team,
    team_vectors=team_vectors,
    statistical_columns=statistical_columns,
    team_id_column=team_id_column,
    )

    home_vector.index = [
        f"home_{column}"
        for column in home_vector.index
    ]

    away_vector.index = [
        f"away_{column}"
        for column in away_vector.index
    ]

    match_vector = pd.concat(
        [
            pd.Series(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                }
            ),
            home_vector,
            away_vector,
        ]
    )

    match_vector = match_vector.to_frame().T

    numeric_columns = [
        column
        for column in match_vector.columns
        if column.startswith(("home_", "away_"))
        and column not in {"home_team", "away_team"}
    ]

    match_vector[numeric_columns] = match_vector[
        numeric_columns
    ].apply(pd.to_numeric, errors="raise")

    return match_vector

def get_team_statistics(df):
    """
    Return the numeric statistics available for both
    home and away teams.

    Identifiers, team names, target variables, and
    match results are excluded.
    """

    excluded_stats = {
        "match_id",
        "team_name",
        "team",
        "score",
        "year",
    }

    home_stats = {
        column.replace("home_", "", 1)
        for column in df.columns
        if column.startswith("home_")
    }

    away_stats = {
        column.replace("away_", "", 1)
        for column in df.columns
        if column.startswith("away_")
    }

    common_stats = sorted(
        home_stats.intersection(away_stats)
    )

    valid_stats = [
        stat
        for stat in common_stats
        if stat not in excluded_stats
        and pd.api.types.is_numeric_dtype(
            df[f"home_{stat}"]
        )
        and pd.api.types.is_numeric_dtype(
            df[f"away_{stat}"]
        )
    ]

    return valid_stats

def create_difference_features(df):
    """
    Create absolute differences between home and away
    team statistics.

    Formula
    -------
    diff_X = home_X - away_X
    """

    df_c = df.copy()

    stats = get_team_statistics(df_c)

    for stat in stats:
        df_c[f"diff_{stat}"] = (
            df_c[f"home_{stat}"]
            - df_c[f"away_{stat}"]
        )

    return df_c

def create_relative_difference_features(
    df,
    epsilon=1e-6,
):
    """
    Create relative differences between home and away
    team statistics.

    Formula
    -------
    relative_diff_X =
        (home_X - away_X)
        / (home_X + away_X + epsilon)
    """

    df_c = df.copy()

    stats = get_team_statistics(df_c)

    for stat in stats:
        home_col = f"home_{stat}"
        away_col = f"away_{stat}"
        new_col = f"relative_diff_{stat}"

        df_c[new_col] = (
            df_c[home_col] - df_c[away_col]
        ) / (
            df_c[home_col]
            + df_c[away_col]
            + epsilon
        )

    return df_c

def create_sum_features(df):
    """
    Create combined sums of home and away
    team statistics.

    Formula
    -------
    sum_X = home_X + away_X
    """

    df_c = df.copy()

    stats = get_team_statistics(df_c)

    for stat in stats:
        home_col = f"home_{stat}"
        away_col = f"away_{stat}"
        new_col = f"sum_{stat}"

        df_c[new_col] = (
            df_c[home_col] + df_c[away_col]
        )

    return df_c

def build_prediction_ready_match_vector(
    home_team,
    away_team,
    team_vectors,
    statistical_columns,
    historical_feature_columns,
    team_id_column="country_clean",
):
    base_vector = build_current_match_vector(
        home_team=home_team,
        away_team=away_team,
        team_vectors=team_vectors,
        statistical_columns=statistical_columns,
        team_id_column=team_id_column,
    )

    home_away_columns = [
        column
        for column in base_vector.columns
        if column.startswith(("home_", "away_"))
        and column not in {"home_team", "away_team"}
    ]

    base_vector[home_away_columns] = (
        base_vector[home_away_columns]
        .astype(float)
    )

    engineered_vector = (
        base_vector
        .pipe(create_difference_features)
        .pipe(create_relative_difference_features)
        .pipe(create_sum_features)
    )

    missing_features = [
        column
        for column in historical_feature_columns
        if column not in engineered_vector.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required historical features: "
            f"{missing_features}"
        )

    return engineered_vector[
        ["home_team", "away_team"]
        + historical_feature_columns
    ].copy()
