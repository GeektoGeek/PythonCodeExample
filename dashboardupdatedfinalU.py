from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go

# Load the primary DataFrame for nTRx Orders and Shipped
df1 = pd.read_csv('C:/Users/17024/Documents/Payneal_Offer/Dashboard_David/DatawithPending12232024U.csv', encoding='iso-8859-1')
df1["Date"] = pd.to_datetime(df1["Date"], errors="coerce")

# Load the second DataFrame for Pending and In_Process
df2 = pd.read_csv('C:/Users/17024/Documents/Payneal_Offer/Dashboard_David/TruePending.csv', encoding='iso-8859-1')
df2["Date"] = pd.to_datetime(df2["WeekStartDate"], errors="coerce")
df2.columns = df2.columns.str.replace(' ', '')

df2.rename(columns={
    "InProcessOrders": "Orders_InProcess",
    "PendingOrders": "Orders_Pending"
}, inplace=True)

# Convert necessary columns to numeric
df2["Orders_InProcess"] = pd.to_numeric(df2["Orders_InProcess"], errors="coerce")
df2["Orders_Pending"] = pd.to_numeric(df2["Orders_Pending"], errors="coerce")

# Initialize Dash app
app = Dash(__name__)

# Color Palette
colors = {
    "ASPN_Orders": "blue", 
    "ASPN_Shipped": "lightblue",
    "CARE_Orders": "green",
    "CARE_Shipped": "lightgreen",
    "PHIL_Orders": "purple",
    "PHIL_Shipped": "violet",
    "Orders_InProcess": "blue",  # Match Orders color
    "Orders_Pending": "lightblue"  # Match Shipped color
}

# Layout of the app
app.layout = html.Div([
    html.H1("Pharmacy Dashboard", style={"textAlign": "center", "margin-bottom": "20px"}),

    # Summary table
    html.Div(id="summary-table", style={"textAlign": "center", "margin-bottom": "20px"}),

    # Dropdown for selecting pharmacies
    html.Div([
        html.Label("Select Pharmacies:", style={"font-weight": "bold"}),
        dcc.Dropdown(
            id="pharmacy-dropdown",
            options=[{"label": pharmacy, "value": pharmacy} for pharmacy in df1["Pharmacies"].unique()],
            value=["ASPN"],  # Default selection
            multi=True,
            clearable=False,
            style={"width": "50%", "margin": "0 auto"}
        )
    ], style={"margin-bottom": "20px"}),

    # Line chart for daily trends
    dcc.Graph(id="line-chart", style={"margin-bottom": "40px"}),

    # Weekly trend chart for In_Process and Pending
    html.Div([
        html.H2("Weekly Trend: In_Process and Pending", style={"textAlign": "center"}),
        dcc.Graph(id="weekly-trend-chart")
    ])
])

# Callback to update the line chart for nTRx Orders and Shipped
@app.callback(
    Output("line-chart", "figure"),
    [Input("pharmacy-dropdown", "value")]
)
def update_line_chart(selected_pharmacies):
    if not selected_pharmacies:  # If no pharmacies are selected, return an empty figure
        return go.Figure()

    filtered_df1 = df1[df1["Pharmacies"].isin(selected_pharmacies)]
    fig = go.Figure()

    # Add traces for Orders_nTRx and Shipped_nTRx for each selected pharmacy
    for pharmacy in selected_pharmacies:
        pharmacy_df = filtered_df1[filtered_df1["Pharmacies"] == pharmacy]
        fig.add_trace(go.Scatter(
            x=pharmacy_df["Date"], 
            y=pharmacy_df["Orders_nTRx"],
            mode="lines+markers",
            name=f"{pharmacy} Orders nTRx",
            line=dict(color=colors.get(f"{pharmacy}_Orders", "black"))
        ))
        fig.add_trace(go.Scatter(
            x=pharmacy_df["Date"], 
            y=pharmacy_df["Shipped_nTRx"],
            mode="lines+markers",
            name=f"{pharmacy} Shipped nTRx",
            line=dict(color=colors.get(f"{pharmacy}_Shipped", "gray"))
        ))

    fig.update_layout(
        title="Daily Trends for Selected Pharmacies",
        xaxis_title="Date",
        yaxis_title="nTRx",
        legend_title="Metric",
        template="plotly_white"
    )
    return fig

# Callback to update the summary table
@app.callback(
    Output("summary-table", "children"),
    [Input("pharmacy-dropdown", "value")]
)
def update_summary_table(selected_pharmacies):
    if not selected_pharmacies:  # If no pharmacies are selected, return a message
        return html.Div("No pharmacies selected.", style={"font-size": "18px", "font-weight": "bold"})

    filtered_df1 = df1[df1["Pharmacies"].isin(selected_pharmacies)]
    total_orders = filtered_df1["Orders_Count"].sum()
    total_shipped = filtered_df1["Shipped_Count"].sum()
    total_orders_ntrx = filtered_df1["Orders_nTRx"].sum()
    total_shipped_ntrx = filtered_df1["Shipped_nTRx"].sum()

    # Calculate the Shipped-to-Orders Ratio
    shipped_to_orders_ratio = (
        total_shipped_ntrx / total_orders_ntrx if total_orders_ntrx > 0 else 0
    )

    summary = html.Table(
        style={"margin": "0 auto", "width": "70%", "border": "1px solid black", "border-collapse": "collapse"},
        children=[
            html.Thead(html.Tr([
                html.Th("Metric (Quarterly)", style={"border": "1px solid black", "padding": "8px"}),
                html.Th("Value", style={"border": "1px solid black", "padding": "8px"})
            ])),
            html.Tbody([
                html.Tr([html.Td("Total Orders", style={"border": "1px solid black", "padding": "8px"}), html.Td(total_orders, style={"border": "1px solid black", "padding": "8px"})]),
                html.Tr([html.Td("Total Shipped", style={"border": "1px solid black", "padding": "8px"}), html.Td(total_shipped, style={"border": "1px solid black", "padding": "8px"})]),
                html.Tr([html.Td("Total Orders nTRx", style={"border": "1px solid black", "padding": "8px"}), html.Td(total_orders_ntrx, style={"border": "1px solid black", "padding": "8px"})]),
                html.Tr([html.Td("Total Shipped nTRx", style={"border": "1px solid black", "padding": "8px"}), html.Td(total_shipped_ntrx, style={"border": "1px solid black", "padding": "8px"})]),
                html.Tr([html.Td("Shipped-to-Orders Ratio (nTRx)", style={"border": "1px solid black", "padding": "8px"}), html.Td(f"{shipped_to_orders_ratio:.2f}", style={"border": "1px solid black", "padding": "8px"})])
            ])
        ]
    )
    return summary

# Callback to update the weekly trend chart
@app.callback(
    Output("weekly-trend-chart", "figure"),
    [Input("pharmacy-dropdown", "value")]
)
def update_weekly_trend_chart(selected_pharmacies):
    if not selected_pharmacies:  # If no pharmacies are selected, return an empty figure
        return go.Figure()

    filtered_df2 = df2[df2["Pharmacy"].isin(selected_pharmacies)]

    # Ensure "Week" column exists
    filtered_df2["Week"] = filtered_df2["Date"].dt.to_period("W").dt.start_time

    # Aggregate by week and pharmacy
    weekly_data = (
        filtered_df2
        .groupby(["Pharmacy", "Week"], as_index=False)
        [["Orders_InProcess", "Orders_Pending"]]
        .sum()
    )

    fig = go.Figure()
    for pharmacy in selected_pharmacies:
        pharmacy_df = weekly_data[weekly_data["Pharmacy"] == pharmacy]
        fig.add_trace(go.Scatter(
            x=pharmacy_df["Week"],
            y=pharmacy_df["Orders_InProcess"],
            mode="lines+markers",
            name=f"{pharmacy} Orders_InProcess",
            line=dict(color=colors.get(f"{pharmacy}_Orders", "blue"))  # Match Orders color
        ))
        fig.add_trace(go.Scatter(
            x=pharmacy_df["Week"],
            y=pharmacy_df["Orders_Pending"],
            mode="lines+markers",
            name=f"{pharmacy} Orders_Pending",
            line=dict(color=colors.get(f"{pharmacy}_Shipped", "lightblue"))  # Match Shipped color
        ))

    fig.update_layout(
        title="Weekly Trends for In Process and Pending",
        xaxis_title="Week",
        yaxis_title="Order Count",
        legend_title="Metric",
        template="plotly_white"
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port=8050, debug=True)

