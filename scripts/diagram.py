from graphviz import Digraph

def create_diagram():
    dot = Digraph(format='png')
    dot.attr(rankdir='LR', size='8,5')

    # Nodes
    dot.node('Company', shape='box')
    dot.node('News Search Task', shape='box')
    dot.node('Price Track Task', shape='box')
    dot.node('Report Analysis Task', shape='box')
    dot.node('Investment Advice Task', shape='box')

    dot.node('News Searcher', shape='box')
    dot.node('Price Tracker', shape='box')
    dot.node('Report Analyst', shape='box')
    dot.node('Investment Advisor', shape='box')

    dot.node('News Search Tool', shape='box')
    dot.node('Stock Price Tool', shape='box')
    dot.node('Data Analysis Tool', shape='box')
    dot.node('Report Retrieval Tool', shape='box')
    dot.node('Report Analysis Tool', shape='box')
    dot.node('Text Format Tool', shape='box')

    # Edges
    dot.edge('Company', 'News Search Task')
    dot.edge('Company', 'Price Track Task')
    dot.edge('Company', 'Report Analysis Task')
    dot.edge('Company', 'Investment Advice Task')

    dot.edge('News Search Task', 'News Searcher')
    dot.edge('Price Track Task', 'Price Tracker')
    dot.edge('Report Analysis Task', 'Report Analyst')
    dot.edge('Investment Advice Task', 'Investment Advisor')

    dot.edge('News Searcher', 'News Search Tool')
    dot.edge('Price Tracker', 'Stock Price Tool')
    dot.edge('Price Tracker', 'Data Analysis Tool')
    dot.edge('Report Analyst', 'Report Retrieval Tool')
    dot.edge('Report Analyst', 'Report Analysis Tool')
    dot.edge('Investment Advisor', 'Text Format Tool')

    return dot

# Create and render the diagram
diagram = create_diagram()
diagram.render('company_tasks_diagram', view=True)