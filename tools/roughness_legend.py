"""Presentation legends separating roughness color from participant symbol."""
import plotly.graph_objects as go


def grouped_legend_positions(experimental_position, row_shift, has_experimental,
                             roughness_x=0.0, participant_x=0.0):
    """Stack legend rows above the experimental anchor when it exists."""
    base_y = experimental_position['y']
    return (
        {'x': roughness_x, 'y': base_y + row_shift * (2 if has_experimental else 1)},
        {'x': participant_x, 'y': base_y + row_shift * int(has_experimental)},
    )


def apply_roughness_participant_legend(
    figure, symbols, participant_symbol_size=16, plot_symbol_size=9,
    roughness_position=None, participant_position=None, experimental_position=None,
    row_shift=0.11, group_token='_roughness_',
):
    groups = {}
    participants = set()
    experimental_traces = []
    for trace in figure.data:
        meta = trace.meta if isinstance(trace.meta, dict) else {}
        pid = meta.get('ipw3_participant_id')
        group = str(trace.legendgroup or '')
        if group.startswith(('experimental_', 'reference_')) or str(trace.name or '').lower().startswith('exp'):
            trace.legend = 'legend3'
            experimental_traces.append(trace)
        if not pid or group_token not in group:
            continue
        pid = str(pid).zfill(3)
        symbol = symbols.get(pid, 'circle')
        trace.marker.symbol = symbol
        trace.marker.size = plot_symbol_size
        trace.mode = 'lines+markers'
        trace.showlegend = False
        groups.setdefault(group, (
            trace.name, trace.line.color, trace.line.dash or 'solid', trace.legendrank or 0,
        ))
        participants.add(pid)
    for rank, (group, (label, color, dash, _)) in enumerate(sorted(groups.items(), key=lambda item: item[1][3])):
        figure.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
            line={'color': color, 'width': 5, 'dash': dash}, name=str(label),
            legendgroup=group, legendrank=rank, showlegend=True, hoverinfo='skip'))
    for rank, pid in enumerate(sorted(participants), start=len(groups)):
        figure.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker={'color': '#000000', 'size': participant_symbol_size, 'symbol': symbols.get(pid, 'circle')},
            name=pid, legend='legend2', legendgroup=f'participant_symbol_{pid}', legendrank=rank,
            showlegend=True, hoverinfo='skip'))
    if participants:
        legend_style = dict(orientation='h', x=0.0, xanchor='left', yanchor='bottom',
                            bgcolor='rgba(0,0,0,0)',
                            font={'size': 24}, traceorder='normal', itemsizing='trace')
        roughness_position = roughness_position or {'x': 0.0, 'y': 1.30}
        participant_position = participant_position or {'x': 0.0, 'y': 1.16}
        experimental_position = experimental_position or {'x': 0.0, 'y': 1.02}
        displayed_roughness_position, displayed_participant_position = grouped_legend_positions(
            experimental_position, row_shift, bool(experimental_traces),
            roughness_x=roughness_position['x'], participant_x=participant_position['x'],
        )
        figure.update_layout(
            legend={**legend_style, **displayed_roughness_position, 'title': {'text': ''}},
            legend2={**legend_style, **displayed_participant_position, 'title': {'text': ''}},
        )
        if experimental_traces:
            figure.update_layout(
                legend3={**legend_style, **experimental_position, 'title': {'text': 'Exp :'}},
                margin={'t': max(300, figure.layout.margin.t or 0)},
            )
        if figure.layout.showlegend is not False:
            minimum_top_margin = 300 if experimental_traces else 220
            figure.update_layout(margin={'t': max(minimum_top_margin, figure.layout.margin.t or 0)})
