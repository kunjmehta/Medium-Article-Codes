import requests as rq
import pandas as pd
import numpy as np
import os
import pulp
import streamlit as st

# The URLs for the FPL API
overview_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
player_url = 'https://fantasy.premierleague.com/api/element-summary/{}/'

'''Get the overall data from the first URL above'''
def get_overview_data():
    overview_data = rq.get(overview_url)
    overview_data_json = overview_data.json()
    return overview_data_json

'''Extract teams data (name, standing, etc) from the overview data'''
def extract_teams_data(overview_data_json):

    # getting the teams dataframe
    teams_json = overview_data_json['teams']
    teams_df = pd.DataFrame(teams_json)
    teams_df = teams_df[['code', 'name', 'short_name', 'position', 'played', 'win', 'draw', \
                         'loss', 'points']]
    teams_df['past_position'] = pd.Series([8, 17, 15, 10, 4, 14, 12, 99, 5, 99, 1, 2, 3, 13, \
                                           9, 11, 6, 16, 99, 7])

    return teams_df

'''Extract the different types of positions of players from the overview data'''
def extract_player_types(overview_data_json):

    # getting player types
    player_types_json = overview_data_json['element_types']
    player_types_df = pd.DataFrame(player_types_json)
    player_types_df = player_types_df[['id', 'singular_name_short', 'squad_select', 'element_count']].\
    rename(columns={'id': 'player_type', 'singular_name_short': 'player_type_code', \
                    'element_count': 'total_players_type'})
    
    return player_types_df


'''Extract the actual player roster and stats from the overview data'''
def extract_player_roster(overview_data_json, player_types_df, teams_df):
    
    # getting player roster
    player_json = overview_data_json['elements']
    player_df = pd.DataFrame(player_json)
    player_df = player_df[['id', 'code', 'first_name', 'second_name', 'web_name', 'team_code',\
                           'element_type', 'status', 'in_dreamteam', 'form', 'total_points', 'points_per_game',\
                           'minutes', 'goals_scored', 'assists', 'clean_sheets',\
                           'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed',\
                           'yellow_cards','red_cards', 'saves', 'bonus', 'now_cost']].\
                            rename(columns={'element_type': 'player_type'})
    # print(player_df['status'].unique()) ['a' 'i' 'u' 'n' 's' 'd']
    # a - available, i - injured, u - sold, n - loan, s - suspended, d - knock
    
    player_df['player_type'] = player_df['player_type'].map(player_types_df.set_index\
                                                            ('player_type')['player_type_code'])
    player_df['team_code'] = player_df['team_code'].map(teams_df.set_index('code')['short_name'])
    
    # print(player_df.dtypes)
    player_df['form'] = player_df['form'].astype(float)
    player_df['points_per_game'] = player_df['points_per_game'].astype(float)
    
    return player_df

'''Make a directory to store the static data'''
def make_dirs():
    if not os.path.exists('./static'):
        os.mkdir('static/')

'''Function to save team data which is static'''
def save_teams_csv(teams_df):
    teams_df.to_csv('static/teams.csv', index = False)

'''Function to save player positions data which is static'''
def save_player_types_csv(player_types_df):
    player_types_df.to_csv('static/player_types.csv', index = False)

'''Function to save player roster data of last season which is static'''
def save_player_csv(player_df):
    player_df.to_csv('static/player_stats_initial.csv', index = False)

'''Get and return the list of injured players for current GW'''
def get_injured_player_list():
    overview_data_json = get_overview_data()
    teams_df = extract_teams_data(overview_data_json)
    player_types_df = extract_player_types(overview_data_json)
    player_df = extract_player_roster(overview_data_json, player_types_df, teams_df)
    injured_players_df = player_df[player_df['status'] != 'a']
    # print('Injured players:', injured_players_df)
    
    return injured_players_df

'''For GW1 and GW4+, get the top scorers for each position, the DT picks'''
def get_top_scorers(top_team_players, positions_filled, teams_filled):
    top_players_point_sort = top_team_players.sort_values(by='total_points', ascending = False)
    
    top_players_positions_to_be_filled = {'GKP':1, 'DEF':1, 'MID': 1, 'FWD': 1}
    top_players_positions_filled = {'GKP':0, 'DEF':0, 'MID': 0, 'FWD': 0}
    team = []
    players_in_team = []
    team_cost = 0
    team_points = 0
    
    i = 0
    while top_players_positions_filled != top_players_positions_to_be_filled and i < \
    len(top_players_point_sort):
        if teams_filled[top_players_point_sort.iloc[i]['team_code']] > 0 \
        and top_players_positions_filled[top_players_point_sort.iloc[i]['player_type']] < 1:
            team.append((top_players_point_sort.iloc[i]['first_name'], \
                        top_players_point_sort.iloc[i]['second_name'],\
                        top_players_point_sort.iloc[i]['player_type'],\
                        top_players_point_sort.iloc[i]['team_code'],\
                        top_players_point_sort.iloc[i]['total_points'],\
                        top_players_point_sort.iloc[i]['now_cost']))
            team_cost += top_players_point_sort.iloc[i]['now_cost']
            team_points += top_players_point_sort.iloc[i]['total_points']
            positions_filled[top_players_point_sort.iloc[i]['player_type']] -= 1
            top_players_positions_filled[top_players_point_sort.iloc[i]['player_type']] += 1
            teams_filled[top_players_point_sort.iloc[i]['team_code']] -= 1
            players_in_team.append(top_players_point_sort.iloc[i]['second_name'])
        
        i += 1
    
    return team, team_cost, team_points, positions_filled, teams_filled, players_in_team


'''For GW2 - 4, get the top scorers for each position, the DT picks'''
def get_top_scorers_for_mixed_data(top_team_players, positions_filled, teams_filled):
    top_players_point_sort = top_team_players.sort_values(by='total_points', ascending = False)
    
    top_players_positions_to_be_filled = {'GKP':1, 'DEF':1, 'MID': 1, 'FWD': 1}
    top_players_positions_filled = {'GKP':0, 'DEF':0, 'MID': 0, 'FWD': 0}
    team = []
    players_in_team = []
    team_cost = 0
    team_points = 0
    
    i = 0
    while top_players_positions_filled != top_players_positions_to_be_filled and i < \
    len(top_players_point_sort):
        if teams_filled[top_players_point_sort.iloc[i]['team_code']] > 0 \
        and top_players_positions_filled[top_players_point_sort.iloc[i]['player_type']] < 1:
            team.append((top_players_point_sort.iloc[i]['first_name'], \
                        top_players_point_sort.iloc[i]['second_name'],\
                        top_players_point_sort.iloc[i]['player_type'],\
                        top_players_point_sort.iloc[i]['team_code'],\
                        top_players_point_sort.iloc[i]['current_points'],\
                        top_players_point_sort.iloc[i]['now_cost']))
            team_cost += top_players_point_sort.iloc[i]['now_cost']
            team_points += top_players_point_sort.iloc[i]['current_points']
            positions_filled[top_players_point_sort.iloc[i]['player_type']] -= 1
            top_players_positions_filled[top_players_point_sort.iloc[i]['player_type']] += 1
            teams_filled[top_players_point_sort.iloc[i]['team_code']] -= 1
            players_in_team.append(top_players_point_sort.iloc[i]['second_name'])
        
        i += 1
    
    return team, team_cost, team_points, positions_filled, teams_filled, players_in_team


'''Define LP problem for selecting other 11 players'''
def add_players_using_lp(metric, costs, player_type, team, budget, team_counts,\
                         positions_filled):
    num_players = len(metric)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    
    # objective function:
    model += sum(decisions[i] * metric[i] for i in range(num_players)), "Objective"

    # cost constraint
    model += sum(decisions[i] * costs[i] for i in range(num_players)) <= budget

    # position constraints
    # 2 total goalkeepers - 1 already selected
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'GKP') == \
    positions_filled['GKP']
    # 5 total defenders - 1 already selected
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'DEF') == \
    positions_filled['DEF']
    # 5 total midfielders - 1 already selected
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'MID') == \
    positions_filled['MID']
    # 3 total attackers - 1 already selected
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'FWD') == \
    positions_filled['FWD']

    # club constraint
    for team_code in np.unique(team):
        model += sum(decisions[i] for i in range(num_players) if team[i] == team_code) <= \
        team_counts[team_code]

    model += sum(decisions) == 11  # total team size

    try:
    	model.solve()
    except:
    	st.info('Player roster has not been updated yet. Please be patient')

    return decisions


'''Define LP problem for selecting player to transfer in'''
def transfer_players_using_lp(metric, costs, player_type, team, budget, team_counts,\
                         positions_filled, num_transfers):
    num_players = len(metric)
    model = pulp.LpProblem("Constrained value maximisation", pulp.LpMaximize)
    decisions = [
        pulp.LpVariable("x{}".format(i), lowBound=0, upBound=1, cat='Integer')
        for i in range(num_players)
    ]
    
    # objective function:
    model += sum(decisions[i] * metric[i] for i in range(num_players)), "Objective"

    # cost constraint
    model += sum(decisions[i] * costs[i] for i in range(num_players)) <= budget

    # position constraints
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'GKP') == \
    2 - positions_filled['GKP']
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'DEF') == \
    5 - positions_filled['DEF']
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'MID') == \
    5 - positions_filled['MID']
    model += sum(decisions[i] for i in range(num_players) if player_type[i] == 'FWD') == \
    3 - positions_filled['FWD']

    # club constraint
    for team_code in np.unique(team):
        model += sum(decisions[i] for i in range(num_players) if team[i] == team_code) <= \
        team_counts[team_code]

    model += sum(decisions) == num_transfers  # total players to be transferred
    try:
    	model.solve()
    except:
    	st.info('Player roster has not been updated yet. Please be patient')
    
    return decisions

'''For GW1, use previous season data to get the team'''
def analyze_using_old_season_data(max_players_from_team, transfer, wildcard, gw, budget):
    player_df = pd.read_csv('static/player_stats_initial.csv')
    teams_df = pd.read_csv('static/teams.csv')
    injured_players_df = get_injured_player_list()
    
    # get list of available players for GW = 1
    available_players = player_df[~player_df['status'].isin(injured_players_df['status'])]
    
    # calculate ROI based on current price and sort on it
    available_players['ROI'] = available_players['total_points']/ (0.1* \
                                                                   available_players['now_cost'])
    available_players = available_players.copy().sort_values(by = ['ROI'], ascending = False)
    
    # Get the last season's top teams' players
    top_teams = teams_df.copy().sort_values(by='past_position')['short_name'][:20]
    top_team_players = available_players[available_players['team_code'].isin(top_teams.tolist())]
    
    # Initialize data structures to track constraints
    positions_filled = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    teams_filled = max_players_from_team
    
    # Add the best point scorers for each position into the team
    predicted_team, team_cost, team_points, positions_filled, teams_filled, players_in_team = \
    get_top_scorers(top_team_players, positions_filled, teams_filled)
    
    # delete selected  top players
    top_team_players = top_team_players[~top_team_players['second_name'].isin(players_in_team)]
    
    # Use LP to find other players based on ROI
    decisions = add_players_using_lp(top_team_players['ROI'].values, \
                                     top_team_players['now_cost'].values, \
                                     top_team_players['player_type'].values, \
                                     top_team_players['team_code'].values, budget - team_cost,\
                                     teams_filled, positions_filled)
    
    for i in range(len(top_team_players)):
        if decisions[i].value() != 0:
            predicted_team.append((top_team_players.iloc[i]['first_name'], \
                                   top_team_players.iloc[i]['second_name'],\
                                   top_team_players.iloc[i]['player_type'],\
                                   top_team_players.iloc[i]['team_code'],\
                                   top_team_players.iloc[i]['total_points'],\
                                   top_team_players.iloc[i]['now_cost']))
            team_cost += top_team_players.iloc[i]['now_cost']
            team_points += top_team_players.iloc[i]['total_points']
            positions_filled[top_team_players.iloc[i]['player_type']] -= 1
            teams_filled[top_team_players.iloc[i]['team_code']] -= 1
            players_in_team.append(top_team_players.iloc[i]['second_name'])
            
    predicted_team = sorted(predicted_team, key = lambda x: (-x[4], x[2]))
    predicted_team = pd.DataFrame.from_records(predicted_team, columns = \
                                              ['First', 'Second', 'Position', 'Team',\
                                               'Points', 'Cost'])
    
    return predicted_team, team_points, team_cost


'''For GW2 - 4, use mix of previous season, current season and form data'''
def analyze_using_mixed_season_data(max_players_from_team, transfer, wildcard, gw, budget, \
                                    old_data_weight, new_data_weight, form_weight, current_team,\
                                    num_transfers):
    
    overview_data_json = get_overview_data()
    mixed_player_df = pd.DataFrame()
    initial_player_df = pd.read_csv('static/player_stats_initial.csv')
    
    # CHANGE THIS DURING DEPLOYMENT. This is dummy data
    # current_player_df = pd.read_csv('player_stats_initial.csv')
    teams_df = extract_teams_data(overview_data_json)
    player_types_df = extract_player_types(overview_data_json)
    current_player_df = extract_player_roster(overview_data_json, player_types_df, teams_df)
    
    # calculate ROI for both current and previous data
    initial_player_df['ROI'] = initial_player_df['total_points']/ (0.1* \
                                       initial_player_df['now_cost'])
    current_player_df['ROI'] = current_player_df['total_points']/ (0.1* \
                                       current_player_df['now_cost'])
    initial_player_df['ROI_scaled'] = (initial_player_df['ROI']- \
                                               initial_player_df['ROI'].min()) / ( \
                                               initial_player_df['ROI'].max() - \
                                               initial_player_df['ROI'].min())
    current_player_df['ROI_scaled'] = (current_player_df['ROI']- \
                                               current_player_df['ROI'].min()) / ( \
                                               current_player_df['ROI'].max() - \
                                               current_player_df['ROI'].min())
    
    # scale 'form' too
    current_player_df['form_scaled'] = (current_player_df['form']- \
                                               current_player_df['form'].min()) / ( \
                                               current_player_df['form'].max() - \
                                               current_player_df['form'].min())
    
    # construct mixed player data after ensuring IDs are aligned
    current_player_df = current_player_df.sort_values(by = ['id'])
    initial_player_df = initial_player_df.sort_values(by = ['id'])
    mixed_player_df = current_player_df.copy(deep = True)
    mixed_player_df['current_points'] = mixed_player_df['total_points']
    
    mixed_player_df['ROI_mixed'] = (old_data_weight * \
                                        initial_player_df['ROI_scaled'] + new_data_weight * \
                                        current_player_df['ROI_scaled'])
    mixed_player_df['ROI_scaled'] = (mixed_player_df['ROI_mixed']- \
                                               mixed_player_df['ROI_mixed'].min()) / ( \
                                               mixed_player_df['ROI_mixed'].max() - \
                                               mixed_player_df['ROI_mixed'].min())
    
    mixed_player_df['metric'] = ((1 - form_weight) * mixed_player_df['ROI_scaled'] + \
                                         (form_weight * current_player_df['form_scaled']))
    mixed_player_df['total_points'] = (initial_player_df['total_points'] * \
                                               old_data_weight) + (\
                                               current_player_df['total_points'] * \
                                               new_data_weight)
    
    injured_players_df = get_injured_player_list()
    
    # get list of available players for GW = gw (2-4)
    initial_available_players = initial_player_df[~initial_player_df['status'].\
                                                  isin(injured_players_df['status'])]
    current_available_players = current_player_df[~current_player_df['status'].\
                                                  isin(injured_players_df['status'])]
    mixed_available_players = mixed_player_df[~mixed_player_df['status'].\
                                                  isin(injured_players_df['status'])]
    # mixed_available_players = mixed_available_players[mixed_available_players['minutes']\
    #                                                      >= 45]

    # mixed_available_players = mixed_available_players.dropna()
    # print(mixed_available_players[mixed_available_players.isna().any(axis=1)])
    # print(len(mixed_available_players), len(current_available_players))

    # giving best performing player irrespective of whether his team plays in the next GW
    if transfer:
        current_team_df = mixed_player_df[mixed_player_df['code']\
                                                    .isin(current_team)]
        
        transfer_out = pd.DataFrame()
        predicted_team = []
        team_points = 0
        team_cost = 0
        positions_filled = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        teams_filled = {key: 3 for key in mixed_available_players['team_code'].unique()}
        
        for i in range(len(current_team_df)):
            if current_team_df.iloc[i]['second_name'] in injured_players_df\
            ['second_name'].tolist():
                transfer_out = transfer_out.append([current_team_df.iloc[i]])
            teams_filled[current_team_df.iloc[i]['team_code']] -= 1
            team_points += current_team_df.iloc[i]['current_points']
            team_cost += current_team_df.iloc[i]['now_cost']
            
        # transfer out injured players as a priority
        if len(transfer_out) > 0:
            transfer_out = transfer_out.copy().sort_values(by = ['metric'], ascending = True)
            transfer_out = transfer_out[:num_transfers]
            for i in range(len(transfer_out)):
                positions_filled[transfer_out.iloc[i]['player_type']] -= 1
                teams_filled[transfer_out.iloc[i]['team_code']] += 1
                team_points -= transfer_out.iloc[i]['current_points']
                team_cost -= transfer_out.iloc[i]['now_cost']
            mixed_available_players = mixed_available_players[~mixed_available_players \
                                        ['second_name'].isin(current_team_df['second_name'].tolist())]
            current_available_players = current_available_players[~current_available_players \
                                        ['second_name'].isin(current_team_df['second_name'].tolist())]
            # mixed_available_players = mixed_available_players.append(transfer_out)
            # print(transfer_out)
        
        # if no injured players, then remove least performing player
        if len(transfer_out) == 0 or num_transfers > len(transfer_out):
            prev_len = len(transfer_out)
            current_team_df = current_team_df.copy().sort_values(by = ['metric'], \
                                                                 ascending = True)
            if len(transfer_out) > 0:
            	current_team_df = current_team_df[~current_team_df \
                                        ['second_name'].isin(transfer_out['second_name'].tolist())]
            # print(current_team_df)
            transfer_out = transfer_out.append(current_team_df[:(num_transfers - len(transfer_out))])
            # print(transfer_out)
            for i in range(prev_len, len(transfer_out)):
                positions_filled[transfer_out.iloc[i]['player_type']] -= 1
                teams_filled[transfer_out.iloc[i]['team_code']] += 1
                team_points -= transfer_out.iloc[i]['current_points']
                team_cost -= transfer_out.iloc[i]['now_cost']
            mixed_available_players = mixed_available_players[~mixed_available_players \
                                        ['second_name'].isin(current_team_df['second_name'].tolist())]
            current_available_players = current_available_players[~current_available_players \
                                        ['second_name'].isin(current_team_df['second_name'].tolist())]
            current_available_players = current_available_players.append(transfer_out)
            
        # here 'budget' is remaining budget from previous week
        decisions = transfer_players_using_lp(mixed_available_players['metric'].values, \
                                         mixed_available_players['now_cost'].values, \
                                         mixed_available_players['player_type'].values, \
                                         mixed_available_players['team_code'].values, \
                                         budget + transfer_out['now_cost'].sum(),\
                                         teams_filled, positions_filled, num_transfers)
        
        for i in range(len(mixed_available_players)):
            if decisions[i].value() != 0:
                predicted_team.append((mixed_available_players.iloc[i]['first_name'], \
                                           mixed_available_players.iloc[i]['second_name'],\
                                           mixed_available_players.iloc[i]['player_type'],\
                                           mixed_available_players.iloc[i]['team_code'],\
                                           mixed_available_players.iloc[i]['current_points'],\
                                           current_available_players.iloc[i]['now_cost']))
                team_cost += mixed_available_players.iloc[i]['now_cost']
                team_points += mixed_available_players.iloc[i]['current_points']
                positions_filled[current_available_players.iloc[i]['player_type']] -= 1
                teams_filled[current_available_players.iloc[i]['team_code']] -= 1

        transfer_out['now_cost'] /= 10
        st.write('Transfer out:', transfer_out[['first_name', 'second_name', 'player_type',\
        										'team_code', 'current_points', 'now_cost']]\
        										.rename(columns = {'first_name':'First Name', \
        										'second_name': 'Second Name', 'player_type': \
        										'Position', 'team_code':'Team',\
        										'current_points':'Points', 'now_cost': 'Cost'}))
        st.write('Transfer in:')

        predicted_team = pd.DataFrame.from_records(predicted_team, columns = \
                                                  ['First', 'Second', 'Position', 'Team',\
                                                   'Points', 'Cost']) 
   
    else:
        # sort on metric
        current_available_players['metric'] = mixed_available_players['metric']
        mixed_available_players = mixed_available_players.copy().sort_values(by = ['metric'], \
                                                                       ascending = False)
        current_available_players = current_available_players.copy().sort_values(by = ['metric'], \
                                                                       ascending = False)

        # Initialize data structures to track constraints
        positions_filled = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        teams_filled = max_players_from_team

        # Add the best point scorers for each position into the team
        predicted_team, team_cost, team_points, positions_filled, teams_filled, players_in_team = \
        get_top_scorers_for_mixed_data(mixed_available_players, positions_filled, teams_filled)

        # delete selected top players
        mixed_available_players = mixed_available_players[~mixed_available_players\
                                                          ['second_name'].\
                                                          isin(players_in_team)]
        current_available_players = current_available_players[~current_available_players\
                                                              ['second_name'].\
                                                              isin(players_in_team)]
        # Just in case rows don't match
        # mixed_available_players = mixed_available_players.dropna()
        # print(mixed_available_players[mixed_available_players['total_points'].isna()])
        
        # Use LP to find other players based on metric
        decisions = add_players_using_lp(mixed_available_players['metric'].values, \
                                         mixed_available_players['now_cost'].values, \
                                         mixed_available_players['player_type'].values, \
                                         mixed_available_players['team_code'].values, \
                                         budget - team_cost, teams_filled, positions_filled)

        for i in range(len(mixed_available_players)):
            if decisions[i].value() != 0:
                predicted_team.append((mixed_available_players.iloc[i]['first_name'], \
                                       mixed_available_players.iloc[i]['second_name'],\
                                       mixed_available_players.iloc[i]['player_type'],\
                                       mixed_available_players.iloc[i]['team_code'],\
                                       mixed_available_players.iloc[i]['current_points'],\
                                       mixed_available_players.iloc[i]['now_cost']))
                team_cost += mixed_available_players.iloc[i]['now_cost']
                team_points += mixed_available_players.iloc[i]['current_points']
                positions_filled[mixed_available_players.iloc[i]['player_type']] -= 1
                teams_filled[mixed_available_players.iloc[i]['team_code']] -= 1
                players_in_team.append(mixed_available_players.iloc[i]['second_name'])

        predicted_team = sorted(predicted_team, key = lambda x: (-x[4], x[2]))
        predicted_team = pd.DataFrame.from_records(predicted_team, columns = \
                                                  ['First', 'Second', 'Position', 'Team',\
                                                   'Points', 'Cost'])
    
    return predicted_team, team_points, team_cost


'''For GW4+, use mix of current season ROI and form data'''
def analyze_using_new_season_data(max_players_from_team, transfer, wildcard, gw, budget, \
                                    form_weight, current_team, num_transfers):
    
    overview_data_json = get_overview_data()
    mixed_player_df = pd.DataFrame()
    
    # CHANGE THIS during deployment. This is dummy data
    # current_player_df = pd.read_csv('player_stats_initial.csv')
    teams_df = extract_teams_data(overview_data_json)
    player_types_df = extract_player_types(overview_data_json)
    current_player_df = extract_player_roster(overview_data_json, player_types_df, teams_df)
    
    # calculate ROI for current data
    current_player_df['ROI'] = current_player_df['total_points']/ (0.1* \
                                       current_player_df['now_cost'])
    current_player_df['ROI_scaled'] = (current_player_df['ROI']- \
                                               current_player_df['ROI'].min()) / ( \
                                               current_player_df['ROI'].max() - \
                                               current_player_df['ROI'].min())
    
    #scale 'form' too
    current_player_df['form_scaled'] = (current_player_df['form']- \
                                               current_player_df['form'].min()) / ( \
                                               current_player_df['form'].max() - \
                                               current_player_df['form'].min())
    
    # calculate metric for current data
    current_player_df['metric'] = (1 - form_weight) * current_player_df['ROI_scaled'] + \
                                          form_weight * current_player_df['form_scaled']
    
    injured_players_df = get_injured_player_list()
    
    # get list of available players for GW = gw 4+
    current_available_players = current_player_df[~current_player_df['status'].\
                                                  isin(injured_players_df['status'])]
    
    # filter those who have played regularly till now (2.5 matches / 4)
    current_available_players = current_available_players[current_available_players['minutes']\
                                                         >= 225]

    # current_available_players = current_available_players.dropna()

    # We give the best transfer option here irrespective of whether that team plays next 
    # week or not. And only for that position
    if transfer:
        current_team_df = current_player_df[current_player_df['code']\
                                                    .isin(current_team)]
        
        transfer_out = pd.DataFrame()
        predicted_team = []
        team_points = 0
        team_cost = 0
        positions_filled = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        teams_filled = {key: 3 for key in current_available_players['team_code'].unique()}
        
        for i in range(len(current_team_df)):
            if current_team_df.iloc[i]['second_name'] in injured_players_df['second_name'].tolist():
                transfer_out = transfer_out.append([current_team_df.iloc[i]])
            teams_filled[current_team_df.iloc[i]['team_code']] -= 1
            team_points += current_team_df.iloc[i]['total_points']
            team_cost += current_team_df.iloc[i]['now_cost']
            
        # transfer out injured players as a priority
        if len(transfer_out) > 0:
            transfer_out = transfer_out.copy().sort_values(by = ['metric'], ascending = True)
            transfer_out = transfer_out[:num_transfers]
            for i in range(len(transfer_out)):
                positions_filled[transfer_out.iloc[i]['player_type']] -= 1
                teams_filled[transfer_out.iloc[i]['team_code']] += 1
                team_points -= transfer_out.iloc[i]['total_points']
                team_cost -= transfer_out.iloc[i]['now_cost']
            current_available_players = current_available_players[~current_available_players \
                                        ['second_name'].isin(current_team_df['second_name'].tolist())]
            # if there are no viable options to transfer
            # current_available_players = current_available_players.append(transfer_out)
        
        # if no injured players, then remove least performing player
        if len(transfer_out) == 0 or num_transfers > len(transfer_out):
            prev_len = len(transfer_out)
            current_team_df = current_team_df.copy().sort_values(by = ['metric'], \
                                                                 ascending = True)
            if len(transfer_out) > 0:
            	current_team_df = current_team_df[~current_team_df \
                                        ['second_name'].isin(transfer_out['second_name'].tolist())]
            # print(current_team_df)
            transfer_out = transfer_out.append(current_team_df[:(num_transfers - len(transfer_out))])
            
            for i in range(prev_len, len(transfer_out)):
                positions_filled[transfer_out.iloc[i]['player_type']] -= 1
                teams_filled[transfer_out.iloc[i]['team_code']] += 1
                team_points -= transfer_out.iloc[i]['total_points']
                team_cost -= transfer_out.iloc[i]['now_cost']
            current_available_players = current_available_players[~current_available_players \
                                        ['second_name'].isin(current_team_df['second_name'].tolist())]
            current_available_players = current_available_players.append(transfer_out)
        
        
        # here 'budget' is remaining budget from previous week
        decisions = transfer_players_using_lp(current_available_players['metric'].values, \
                                         current_available_players['now_cost'].values, \
                                         current_available_players['player_type'].values, \
                                         current_available_players['team_code'].values, \
                                         budget + transfer_out['now_cost'].sum(), teams_filled, \
                                                  positions_filled, num_transfers)
            
        for i in range(len(current_available_players)):
            if decisions[i].value() != 0:
                predicted_team.append((current_available_players.iloc[i]['first_name'], \
                                           current_available_players.iloc[i]['second_name'],\
                                           current_available_players.iloc[i]['player_type'],\
                                           current_available_players.iloc[i]['team_code'],\
                                           current_available_players.iloc[i]['total_points'],\
                                           current_available_players.iloc[i]['now_cost']))
                team_cost += current_available_players.iloc[i]['now_cost']
                team_points += current_available_players.iloc[i]['total_points']
                positions_filled[current_available_players.iloc[i]['player_type']] -= 1
                teams_filled[current_available_players.iloc[i]['team_code']] -= 1

        transfer_out['now_cost'] /= 10
        st.write('Transfer out:', transfer_out[['first_name', 'second_name', 'player_type',\
        										'team_code', 'total_points', 'now_cost']]\
        										.rename(columns = {'first_name':'First Name', \
        										'second_name': 'Second Name', 'player_type': \
        										'Position', 'team_code':'Team',\
        										'total_points':'Points', 'now_cost': 'Cost'}))
        st.write('Transfer in:')

        predicted_team = pd.DataFrame.from_records(predicted_team, columns = \
                                                  ['First', 'Second', 'Position', 'Team',\
                                                   'Points', 'Cost']) 

    # We give the option of inputting max_players from a team as 0 if the team is not playing
    # next week
    else:
        # sort on metric
        current_available_players = current_available_players.copy().sort_values(by = ['metric'], \
                                                                       ascending = False)

        # Initialize data structures to track constraints
        positions_filled = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        teams_filled = max_players_from_team

        # Add the best point scorers for each position into the team
        predicted_team, team_cost, team_points, positions_filled, teams_filled, players_in_team = \
        get_top_scorers(current_available_players, positions_filled, teams_filled)

        # delete selected top players
        current_available_players = current_available_players[~current_available_players['second_name'].\
                                                isin(players_in_team)]

        # Use LP to find other players based on ROI
        decisions = add_players_using_lp(current_available_players['metric'].values, \
                                         current_available_players['now_cost'].values, \
                                         current_available_players['player_type'].values, \
                                         current_available_players['team_code'].values, \
                                         budget - team_cost, teams_filled, positions_filled)

        for i in range(len(current_available_players)):
            if decisions[i].value() != 0:
                predicted_team.append((current_available_players.iloc[i]['first_name'], \
                                       current_available_players.iloc[i]['second_name'],\
                                       current_available_players.iloc[i]['player_type'],\
                                       current_available_players.iloc[i]['team_code'],\
                                       current_available_players.iloc[i]['total_points'],\
                                       current_available_players.iloc[i]['now_cost']))
                team_cost += current_available_players.iloc[i]['now_cost']
                team_points += current_available_players.iloc[i]['total_points']
                positions_filled[current_available_players.iloc[i]['player_type']] -= 1
                teams_filled[current_available_players.iloc[i]['team_code']] -= 1
                players_in_team.append(current_available_players.iloc[i]['second_name'])

        predicted_team = sorted(predicted_team, key = lambda x: (-x[4], x[2]))
        predicted_team = pd.DataFrame.from_records(predicted_team, columns = \
                                                  ['First', 'Second', 'Position', 'Team',\
                                                   'Points', 'Cost'])    
    
    return predicted_team, team_points, team_cost


'''Main predict function that branches to computation as per hyperparameters'''
# For testing, all params are set. During deployment, take this from the website
def predict_team(transfer = False, wildcard = False, gw = 1, \
                 budget = 1000, old_data_weight = 0.4, new_data_weight = 0.6, form_weight = 0.5, \
                 max_players_from_team = None, current_team = None, num_transfers = 1):
    if gw > 4:
        long_form, points, team_cost = analyze_using_new_season_data(max_players_from_team, \
                                                                     transfer, wildcard, gw, \
                                                                     budget, form_weight, \
                                                                     current_team, num_transfers)
        return long_form, points, team_cost
    elif gw > 1 and gw <= 4:
        long_form, points, team_cost = analyze_using_mixed_season_data(max_players_from_team,\
                                                                       transfer, wildcard, gw,\
                                        budget, old_data_weight, new_data_weight, form_weight,\
                                                                       current_team, num_transfers)
        return long_form, points, team_cost
    elif gw == 1:
        long_form, points, team_cost = analyze_using_old_season_data(\
                                                    max_players_from_team, \
                                                    transfer, wildcard, gw, budget)
        return long_form, points, team_cost


# if name == '__main__()':
# 	overview_data_json = get_overview_data()
# 	teams_df = extract_teams_data(overview_data_json)
# 	player_types_df = extract_player_types(overview_data_json)
# 	player_df = extract_player_roster(overview_data_json)
# 	make_dirs()
# 	save_teams_csv(teams_df)
# 	save_player_types_csv(player_types_df)
# 	save_player_csv(player_df)
# 	predict_team(transfer = False, wildcard = False, gw = 1, \
#                  budget = 1000, old_data_weight = 0.4, new_data_weight = 0.6, form_weight = 0.5, \
#                  max_players_from_team = None, current_team = None, num_transfers = 1)