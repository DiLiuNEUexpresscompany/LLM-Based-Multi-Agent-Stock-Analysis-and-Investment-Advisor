from graphviz import Digraph

def create_diagram():
    dot = Digraph(format='svg')  # Changed format from 'png' to 'svg'
    dot.attr(rankdir='LR', size='8,5')
    dot.attr(bgcolor='#FFFFFF')  # Set light gray background
    
    # Color scheme
    corporate_blue = '#2A5CAA'
    accent_green = '#82B366'
    accent_orange = '#D6B656'
    accent_red = '#B85450'
    accent_purple = '#9673A6'
    
    # Common node attributes
    dot.attr('node', style='filled', shape='box', 
            fontname='Helvetica', fontsize='12', 
            penwidth='1.5', margin='0.3')
    
    # Main company node
    dot.node('Company', label='Company\n(Investment Analysis)', 
            fillcolor=corporate_blue, fontcolor='white',
            shape='folder', width='2', height='1')

    # Task nodes
    tasks = {
        'News Search Task': accent_green,
        'Price Track Task': accent_orange,
        'Report Analysis Task': accent_red,
        'Investment Advice Task': accent_purple
    }
    
    for task, color in tasks.items():
        dot.node(task, fillcolor=color, fontcolor='white',
                width='1.8', height='0.8')

    # Role nodes
    roles = {
        'News Searcher': '#A9D08E',
        'Price Tracker': '#FFD966',
        'Report Analyst': '#EA9999',
        'Investment Advisor': '#CAB1D6'
    }
    
    for role, color in roles.items():
        dot.node(role, fillcolor=color, fontcolor='black',
                shape='box3d', width='1.6', height='0.7')

    # Tool nodes
    tools = {
        'News Search Tool': '#BDD7EE',
        'Stock Price Tool': '#F8CBAD',
        'Data Analysis Tool': '#E2EFDA',
        'Report Retrieval Tool': '#F4CCCC',
        'Report Analysis Tool': '#D9E1F2',
        'Text Format Tool': '#FFF2CC'
    }
    
    for tool, color in tools.items():
        dot.node(tool, fillcolor=color, fontcolor='black',
                shape='cylinder', width='1.5', height='0.6')

    # Edges
    dot.edge('Company', 'News Search Task', color=corporate_blue, penwidth='2')
    dot.edge('Company', 'Price Track Task', color=corporate_blue, penwidth='2')
    dot.edge('Company', 'Report Analysis Task', color=corporate_blue, penwidth='2')
    dot.edge('Company', 'Investment Advice Task', color=corporate_blue, penwidth='2')

    task_to_role = [
        ('News Search Task', 'News Searcher'),
        ('Price Track Task', 'Price Tracker'),
        ('Report Analysis Task', 'Report Analyst'),
        ('Investment Advice Task', 'Investment Advisor')
    ]
    
    for task, role in task_to_role:
        dot.edge(task, role, color='#666666', style='dashed', penwidth='1.5')

    dot.edge('News Searcher', 'News Search Tool', color='#5B9BD5')
    dot.edge('Price Tracker', 'Stock Price Tool', color='#ED7D31')
    dot.edge('Price Tracker', 'Data Analysis Tool', color='#ED7D31')
    dot.edge('Report Analyst', 'Report Retrieval Tool', color='#FF5050')
    dot.edge('Report Analyst', 'Report Analysis Tool', color='#FF5050')
    dot.edge('Investment Advisor', 'Text Format Tool', color='#7030A0')

    return dot

# Create and save the diagram
diagram = create_diagram()
diagram.render('company_tasks_diagram', format='svg', cleanup=True)  # Added format parameter and cleanup flag