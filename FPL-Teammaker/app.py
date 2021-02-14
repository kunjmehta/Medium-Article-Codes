import streamlit as st
import pandas as pd
from fpl import predict_team, get_overview_data, extract_player_roster, \
extract_teams_data, extract_player_types
import matplotlib.pyplot as plt
import numpy as np

pd.options.display.float_format = "{:,.2f}".format

def get_team_limit(max_players_from_team):
	max_players_from_team['ARS'] = int(st.text_input('ARS:', 3))
	max_players_from_team['AVL'] = int(st.text_input('AVL:', 3))
	max_players_from_team['BHA'] = int(st.text_input('BHA:', 3))
	max_players_from_team['BUR'] = int(st.text_input('BUR:', 3))
	max_players_from_team['CHE'] = int(st.text_input('CHE:', 3))
	max_players_from_team['CRY'] = int(st.text_input('CRY:', 3))
	max_players_from_team['EVE'] = int(st.text_input('EVE:', 3))
	max_players_from_team['FUL'] = int(st.text_input('FUL:', 3))
	max_players_from_team['LEE'] = int(st.text_input('LEE:', 3))
	max_players_from_team['LEI'] = int(st.text_input('LEI:', 3))
	max_players_from_team['LIV'] = int(st.text_input('LIV:', 3))
	max_players_from_team['MCI'] = int(st.text_input('MCI:', 3))
	max_players_from_team['MUN'] = int(st.text_input('MUN:', 3))
	max_players_from_team['NEW'] = int(st.text_input('NEW:', 3))
	max_players_from_team['SHU'] = int(st.text_input('SHU:', 3))
	max_players_from_team['SOU'] = int(st.text_input('SOU:', 3))
	max_players_from_team['TOT'] = int(st.text_input('TOT:', 3))
	max_players_from_team['WBA'] = int(st.text_input('WBA:', 3))
	max_players_from_team['WHU'] = int(st.text_input('WHU:', 3))
	max_players_from_team['WOL'] = int(st.text_input('WOL:', 3))

	return max_players_from_team


st.markdown("<h1 style='text-align: center;'>Welcome to FPL TeamMaker</h1>", \
			unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Use Data Science to build your \
			team and win!</h3>", unsafe_allow_html=True)

transfer = False
wildcard = False
gw = 1
budget = 1000
old_data_weight = 0.4
new_data_weight = 0.6
form_weight = 0.5
max_players_from_team = {}
current_team = []
num_transfers = 1


gw = int(st.text_input('Enter the Gameweek you want to make team for:', '1'))
if gw == 1:
	st.write('Starting below, please provide how many players you want from each team.\
		Use this in cases when a particular team does not have a fixture for the week.')
	max_players_from_team = get_team_limit(max_players_from_team)

elif gw > 1 and gw <= 4:
	transfer_or_wildcard = st.radio('Select your mode of team making:', ('Transfer',\
								'New Team / Wildcard'))
	if transfer_or_wildcard == 'Transfer':
		transfer = True
	else:
		wildcard = True

	old_data_weight = float(st.text_input('Enter the weight you want to give to last \
								season\'s  data (0-1.0):', 0.4))
	new_data_weight = float(st.text_input('Enter the weight you want to give to current \
								season\'s  data (0-1.0):', 0.6))
	form_weight = float(st.text_input('Enter the weight you want to give to player form \
							(0-1.0):', 0.5))
	budget = float(st.text_input('Enter your budget x 10 (For transfers, enter \
								 the leftover budget using current team):', 1000))

	if transfer:
		num_transfers = int(st.text_input('Enter the number of transfers to be made:', 1))
		overview_data_json = get_overview_data()
		teams_df = extract_teams_data(overview_data_json)
		player_types_df = extract_player_types(overview_data_json)
		player_df = extract_player_roster(overview_data_json, player_types_df, teams_df)
		player_df = player_df[['code', 'first_name', 'second_name', 'team_code']]
		players = st.write('Please look at the list below and enter a comma \
								 separated list of player codes you have in your team. \
								 Note that they are ordered alphabetically by team name.', \
								 player_df)
		try:
			current_team = st.text_input('')
			current_team = list(map(int, current_team.split(',')))
		except:
			st.error('Please enter an input above')

	else:
		st.write('Starting below, please provide how many players you want from each team.\
		Use this in cases when a particular team does not have a fixture for the week.')
		max_players_from_team = get_team_limit(max_players_from_team)

elif gw > 4 and gw <=38:
	transfer_or_wildcard = st.radio('Select your mode of team making:', ('Transfer',\
								'New Team / Wildcard'))
	if transfer_or_wildcard == 'Transfer':
		transfer = True
	else:
		wildcard = True


	form_weight = float(st.text_input('Enter the weight you want to give to player form \
							(0-1.0):', 0.5))
	budget = float(st.text_input('Enter your budget x 10 (For transfers, enter \
								 the leftover budget using current team):', 1000))

	if transfer:
		num_transfers = int(st.text_input('Enter the number of transfers to be made:', 1))
		overview_data_json = get_overview_data()
		teams_df = extract_teams_data(overview_data_json)
		player_types_df = extract_player_types(overview_data_json)
		player_df = extract_player_roster(overview_data_json, player_types_df, teams_df)
		player_df = player_df[['code', 'first_name', 'second_name', 'team_code']]
		players = st.write('Please look at the list below and enter a comma \
								 separated list of player codes you have in your team. \
								 Note that they are ordered alphabetically by team name.', \
								 player_df)
		try:
			current_team = st.text_input('')
			current_team = list(map(int, current_team.split(',')))
		except:
			st.error('Please enter an input above')

	else:
		st.write('Starting below, please provide how many players you want from each team.\
		Use this in cases when a particular team does not have a fixture for the week.')
		max_players_from_team = get_team_limit(max_players_from_team)

if st.button('Get Team'):

	if gw > 38:
		st.error('Enter correct GW')

	team, points, cost = predict_team(transfer, wildcard, gw, budget, old_data_weight, \
				 new_data_weight, form_weight, max_players_from_team, \
				 current_team, num_transfers)
	team['Cost'] /= 10
	team = team.rename(columns = {"First": "First Name", "Second": "Second Name"})
	if len(team) > 0:
		st.write(team)
		st.write('Total points of whole team:', points)
		st.write('Cost of the team:', cost)
	else:
		st.info('Please use this feature after GW4 has completed')


st.markdown("<h1 style='text-align: center;'>Visualization of Results</h1>", unsafe_allow_html=True)


bot101_points = np.array([0, 84, 133, 180, 222, 294, 349, 401, 470\
, 551, 593, 662, 723, 774, 866, 914, 965, 1028, 1070\
,  1151, 1196, 1268, 1352, 1403, np.nan, np.nan, np.nan, np.nan\
, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

average_points = np.array([0, 50, 109, 152, 200, 260, 308, 361, 416\
, 471, 515, 577, 628, 670, 730, 771, 808, 864, 894\
, 968, 1010, 1058, 1115, 1173, np.nan, np.nan, np.nan, np.nan, np.nan\
, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

bot101_rank = np.array([0, 0.175178, 0.881887, 0.797683, 1.412642, 1.056323, 0.915314, 1.0164, 0.785028\
, 0.378354, 0.515661, 0.522446, 0.495127, 0.426231, 0.218577, 0.226317, 0.191098, 0.204354, 0.219032\
, 0.302307, 0.318217, 0.224458, 0.183696, 0.275990, np.nan, np.nan, np.nan, np.nan, np.nan\
, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

bot101_gwrank = np.array([0, 0.175178, 4.534625, 1.174666, 4.348521, 1.728420, 2.062787, 3.960282, 1.105014\
, 0.247396, 2.6096, 2.559066, 1.815474, 1.603100, 0.178608, 2.406347, 0.969028, 2.743019, 1.780634\
, 2.626326, 2.934196, 0.699079, 0.497828, 5.061059, np.nan, np.nan, np.nan, np.nan, np.nan\
, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

bot101_gw_points = np.array([0, 84, 49, 55, 42, 72, 55, 52, 69, 81, 50, 69, 61, 51, 92, 48, 51, 63, \
42, 81, 45, 72, 88, 55, np.nan, np.nan, np.nan, np.nan, np.nan\
, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

bot101_gw_avg_points = np.array([0, 50, 59, 43, 48, 60, 48, 53, 55, 55, 44, 62, 51, 42, 60, 41, 37, \
56, 30, 74, 42, 48, 57, 58, np.nan, np.nan, np.nan, np.nan, np.nan\
, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

gameweeks = [i for i in range(0, 39)]

ax = plt.gca()
ax.set_xlim(0,39)
plt.plot(gameweeks, bot101_points, label = 'Cumulative Team Points')
plt.plot(gameweeks, average_points, label = 'Cumulative Average Points')
plt.xlabel('Gameweeks')
plt.ylabel('Total Points')
plt.title('Total Points Viewed per Week')
plt.legend()
plt.locator_params(axis="x", nbins=38)
st.pyplot()

ax = plt.gca()
ax.invert_yaxis()
ax.ticklabel_format(useOffset=False, style = 'plain') 
plt.locator_params(axis="y", nbins=20)
plt.locator_params(axis="x", nbins=38)

ax.set_xlim(0,39)
plt.plot(gameweeks, bot101_rank, label = 'Overall Rank')
plt.plot(gameweeks, bot101_gwrank, label = 'GW Rank')
plt.xlabel('Gameweeks')
plt.ylabel('Ranking (in millions)')
plt.title('Overall vs GW Rank')
plt.legend()
st.pyplot()

ax = plt.gca()
ax.set_xlim(0,39)
plt.plot(gameweeks, bot101_gw_points, label = 'Team Points per Gameweek')
plt.plot(gameweeks, bot101_gw_avg_points, label = 'Average Points per Gameweek')
plt.xlabel('Gameweeks')
plt.ylabel('Points')
plt.title('Points per Week')
plt.legend()
plt.locator_params(axis="x", nbins=38)
st.pyplot()