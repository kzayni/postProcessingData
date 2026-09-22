"""Participant panels built from styled roughness comparison curves."""
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_roughness_participant_panels(
    source, *, ice_shapes=False, columns=None, participant_title_size=30,
    participant_model_labels=None,
):
    curves = {}
    for trace in source.data:
        meta = trace.meta if isinstance(trace.meta, dict) else {}
        pid = meta.get('ipw3_participant_id')
        if pid and '_roughness_' in str(trace.legendgroup):
            curves.setdefault(str(pid).zfill(3), []).append(trace)
    if not curves:
        raise ValueError('No participant roughness curves available')
    ids = sorted(curves)
    if columns is not None:
        cols = min(int(columns), len(ids))
    else:
        cols = (2 if len(ids) <= 4 else 3) if ice_shapes else 3
        cols = min(cols, len(ids)) if ice_shapes else cols
    rows = math.ceil(len(ids) / cols)
    positions = [(index // cols + 1, index % cols + 1) for index in range(len(ids))]
    if len(ids) % cols == 1 and cols == 3:
        positions[-1] = (rows, 2)
    titles = [''] * (rows * cols)
    for pid, (row, col) in zip(ids, positions):
        model = (participant_model_labels or {}).get(pid)
        titles[(row - 1) * cols + col - 1] = f'Participant {pid} ({model})' if model else f'Participant {pid}'
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        horizontal_spacing=0.06 if ice_shapes else 0.09, vertical_spacing=0.24 / rows if ice_shapes else 0.32 / rows)
    first_row, first_col = positions[0]
    first_axis_number = (first_row - 1) * cols + first_col
    first_axis_suffix = str(first_axis_number) if first_axis_number > 1 else ''
    groups = set()
    for index, pid in enumerate(ids):
        row, col = positions[index]
        if ice_shapes:
            for original in source.data:
                if original.legendgroup == 'clean_reference':
                    clean = go.Scatter(original.to_plotly_json())
                    clean.update(xaxis=None, yaxis=None, showlegend=False)
                    fig.add_trace(clean, row=row, col=col)
        for original in curves[pid]:
            trace = go.Scatter(original.to_plotly_json())
            trace.update(xaxis=None, yaxis=None, legend='legend', showlegend=False)
            fig.add_trace(trace, row=row, col=col)
            if trace.legendgroup not in groups:
                groups.add(trace.legendgroup)
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                    line={'color': trace.line.color, 'width': 5}, name=trace.name,
                    legendgroup=trace.legendgroup, legendrank=trace.legendrank,
                    showlegend=True, hoverinfo='skip'), row=row, col=col)
        for name in ('xaxis', 'yaxis'):
            settings = getattr(source.layout, name).to_plotly_json()
            for key in ('domain', 'anchor', 'overlaying', 'matches', 'scaleanchor'):
                settings.pop(key, None)
            settings['title'] = {'text': getattr(source.layout, name).title.text, 'font': {'size': 20}}
            settings['tickfont'] = {'size': 16}
            settings['automargin'] = True
            # Match only populated panels. Plotly then autoranges over all
            # participant curves together, while retaining fixed source limits.
            if (row, col) != positions[0]:
                settings['matches'] = name[0] + first_axis_suffix
            update = fig.update_xaxes if name == 'xaxis' else fig.update_yaxes
            update(**settings, row=row, col=col)
        if any(shape.type == 'line' and shape.y0 == 0 and shape.y1 == 0 for shape in source.layout.shapes):
            fig.add_hline(y=0, line_dash='dash', line_color='black', row=row, col=col)
    for index in range(rows * cols):
        if (index // cols + 1, index % cols + 1) in positions:
            continue
        fig.update_xaxes(visible=False, row=index // cols + 1, col=index % cols + 1)
        fig.update_yaxes(visible=False, row=index // cols + 1, col=index % cols + 1)
    if cols == 2 and len(ids) % 2 == 1:
        # Center a lone participant in the last row while retaining the same
        # panel width as the complete rows above it.
        last_row, last_col = positions[-1]
        axis_number = (last_row - 1) * cols + last_col
        suffix = str(axis_number) if axis_number > 1 else ''
        xaxis = fig.layout[f'xaxis{suffix}']
        panel_width = xaxis.domain[1] - xaxis.domain[0]
        xaxis.domain = [0.5 - panel_width / 2, 0.5 + panel_width / 2]
        for annotation in fig.layout.annotations:
            if annotation.text == titles[(last_row - 1) * cols + last_col - 1]:
                annotation.x = 0.5
                break
    fig.update_layout(width=1800, height=rows * 470 + 150,
        paper_bgcolor='white', plot_bgcolor='white', font={'family': 'Arial', 'size': 20},
        margin={'l': 90, 'r': 45, 't': 150, 'b': 70},
        legend={'orientation': 'h', 'x': 0, 'y': 1.04, 'yanchor': 'bottom',
                'title': {'text': ''}, 'font': {'size': 24}, 'traceorder': 'normal'})
    fig.update_annotations(font=dict(size=participant_title_size), yshift=8)
    if ice_shapes:
        x_range, y_range = source.layout.xaxis.range, source.layout.yaxis.range
        width = 1800
        inner_width = width - 130
        panel_width = inner_width * (1 - 0.06 * (cols - 1)) / cols
        panel_height = panel_width * abs((y_range[1] - y_range[0]) / (x_range[1] - x_range[0]))
        height = round(rows * panel_height / (1 - (0.24 / rows) * (rows - 1)) + 190)
        fig.update_layout(width=width, height=height, margin=dict(l=95, r=35, t=115, b=75))
        fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=20))
        fig.update_yaxes(title_font=dict(size=24), tickfont=dict(size=20))
        for row, col in positions:
            if row < rows:
                fig.update_xaxes(title_text='', row=row, col=col)
            if col > 1:
                fig.update_yaxes(title_text='', row=row, col=col)
    return fig
