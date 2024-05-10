import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List

import conf as conf

def generate_random_candidates_positions(num_candidates: int) -> np.ndarray:
    while True:
        positions = np.unique(np.random.uniform(
            low=conf.MIN_POSITION, high=conf.MAX_POSITION, size=num_candidates
        ))
        positions.sort()
        if len(positions) == num_candidates:
            return positions

def parse_candidates_positions(
    candidates_positions: str, num_candidates: int
) -> List[float]:
    positions = [float(e.strip()) for e in candidates_positions.split(',')]
    if len(positions) != num_candidates:
        raise Exception('Positions do not match number of candidates')
    positions.sort()
    return positions

def parse_scoring_rule(scoring_rule: str, num_candidates: int) -> List[float]:
    scores = [float(e.strip()) for e in scoring_rule.split(',')]

    if len(scores) > num_candidates:
        scores = scores[:num_candidates]
    if len(scoring_rule) < num_candidates:
        scores += [0 for _ in range(num_candidates - len(scores))]
    
    return scores

def parse_voting_profile(voting_profile: str, st) -> List[List[float]]:
    if voting_profile[0] != '[' or voting_profile[-1] != ']':
        raise Exception('Invalid format')
    voting_profile = voting_profile.replace(' ', '')[1:-1]
    return [
        [float(e) for e in voter.split(',')]
        for voter in voting_profile.split('],[')
    ]

def plot_positions(
    candidates_pd: pd.DataFrame, voters_pd: pd.DataFrame
) -> go.Figure:
    positions_plot = go.Figure()

    positions_plot.add_trace(go.Scatter(
        x=candidates_pd[conf.POSITION],
        y=[0]*len(candidates_pd),
        text=candidates_pd[conf.ID],
        textposition='bottom center',
        name=conf.CANDIDATE,
        mode='markers+text'
    ))

    positions_plot.add_trace(go.Scatter(
        x=voters_pd[conf.POSITION],
        y=[0]*len(voters_pd),
        text=voters_pd[conf.ID],
        textposition='top center',
        name=conf.VOTER,
        mode='markers+text'
    ))

    positions_plot.update_layout(height=220)
    return positions_plot

def plot_scores(
    candidates_pd: pd.DataFrame, voters_pd: pd.DataFrame, scoring_rule_pd: pd.DataFrame
) -> go.Figure:
    ranks_pd = (
        voters_pd
        .rename(columns={conf.ID: conf.VOTER, conf.POSITION: conf.VOTER_POSITION})
        .merge(
            candidates_pd.rename(columns={
                conf.ID: conf.CANDIDATE, conf.POSITION: conf.CANDIDATE_POSITION
            }),
            how='cross'
        )
    )
    ranks_pd[conf.DISTANCE] = (
        ranks_pd[conf.CANDIDATE_POSITION] - ranks_pd[conf.VOTER_POSITION]
    ).abs()
    ranks_pd[conf.RANK] = (
        ranks_pd.groupby(conf.VOTER)[conf.DISTANCE].rank(method='first') - 1
    ).astype(int)

    scores_pd = ranks_pd.merge(
        scoring_rule_pd, left_on=conf.RANK, right_index=True
    )
    fig = px.bar(
        scores_pd, x=conf.CANDIDATE, y=conf.SCORE, color=conf.VOTER,
    )
    fig.update_traces(marker_line_color='white', marker_line_width=1)
    return fig


st.title('Spatial Voting')

# Profile input
num_candidates = int(st.number_input(
    'Number of candidates', min_value=2, value=2
))
candidates_positions_option = st.selectbox(
    'Candidates positions', options=[conf.UNIFORM, conf.RANDOM, conf.CUSTOM]
)
if candidates_positions_option == conf.UNIFORM:
    candidates_positions = np.linspace(
        conf.MIN_POSITION, conf.MAX_POSITION, num=num_candidates
    )
elif candidates_positions_option == conf.RANDOM:
    candidates_positions = generate_random_candidates_positions(num_candidates)
else:
    candidates_positions_string = st.text_input(
        '|C| positions, separated by commas',
        value=f'{conf.MIN_POSITION}, {conf.MAX_POSITION}'
    )
    candidates_positions = parse_candidates_positions(
        candidates_positions_string, num_candidates
    )

scoring_rule_string = st.text_input(
    'Scoring rule: |C| scores, separated by commas. Missing entries are filled with zeros.',
    value='1'
)
scoring_rule = parse_scoring_rule(scoring_rule_string, num_candidates)

voting_profile_string = st.text_input(
    'Voting profile: [l1, u1], [l2, u2], ... (pair for each voter, separated by commas)',
    value=f'[{conf.MIN_POSITION}, {conf.MAX_POSITION}]'
)
voting_profile = parse_voting_profile(voting_profile_string, st)

# Voter positions input
voters_positions = [
    st.slider(f'Select a position for v{i}', start, end, start)
    for i, (start, end) in enumerate(voting_profile)
]

# plots
candidates_pd = pd.DataFrame({
    conf.ID: [f'c{i}' for i in range(num_candidates)],
    conf.POSITION: candidates_positions
})
voters_pd = pd.DataFrame({
    conf.ID: [f'v{i}' for i in range(len(voters_positions))],
    conf.POSITION: voters_positions
})
scoring_rule_pd = pd.DataFrame({conf.SCORE: scoring_rule})

st.plotly_chart(
    plot_positions(candidates_pd, voters_pd), use_container_width=True
)
st.plotly_chart(
    plot_scores(candidates_pd, voters_pd, scoring_rule_pd), use_container_width=True
)
