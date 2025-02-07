from graphviz import Digraph

def create_diagram():
    dot = Digraph(format='png')
    dot.attr(rankdir='TD', size='20,30')
    # Light blue-gray background
    dot.attr(bgcolor='#FFFFFF')
    
    # Blue-based color scheme
    primary_blue = '#1A365D'  # Dark blue for main node
    task_blue = '#2B4C7E'     # Medium blue for tasks
    agent_blue = '#4A6FA5'    # Lighter blue for agents
    tool_blue = '#7B9ED9'     # Lightest blue for tools
    
    # Common node attributes with shadow effect
    dot.attr('node', style='filled,shadowed', shape='box', 
            fontname='Arial', fontsize='12', 
            penwidth='2', margin='0.4')
    
    # Edge attributes
    dot.attr('edge', fontname='Arial', fontsize='10', 
            penwidth='1.5', arrowsize='0.8')

    # Legend with blue theme
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='rounded,filled', 
                   color=primary_blue, fillcolor='#EDF2F7', 
                   fontcolor=primary_blue)
        legend.node('legend_task', 'Task Node', shape='box', 
                   fillcolor=task_blue, fontcolor='white')
        legend.node('legend_agent', 'Agent Node', shape='box3d', 
                   fillcolor=agent_blue, fontcolor='white')
        legend.node('legend_tool', 'Tool Node', shape='cylinder', 
                   fillcolor=tool_blue, fontcolor='white')
        legend.attr(rank='same')
    
    # Main company node
    dot.node('Company', label='Investment Analysis System', 
            fillcolor=primary_blue, fontcolor='white',
            shape='folder', width='2.5', height='1.2')

    # Task nodes with gradient-like effect using different blue shades
    tasks = {
        'News Search Task': ('News Search\nSubroutine', task_blue),
        'Price Track Task': ('Price Tracking\nSubroutine', '#2F5288'),
        'Report Analysis Task': ('Report Analysis\nSubroutine', '#335892'),
        'Investment Advice Task': ('Investment Advisory\nSubroutine', '#385E9C')
    }
    
    for task_id, (label, color) in tasks.items():
        dot.node(task_id, label=f'Task: {label}', 
                fillcolor=color, fontcolor='white',
                width='2', height='1')

    # Role nodes with lighter blue shades
    roles = {
        'News Searcher': ('News Search\nAgent', agent_blue),
        'Price Tracker': ('Price Tracking\nAgent', '#547AB0'),
        'Report Analyst': ('Report Analysis\nAgent', '#5E84BA'),
        'Investment Advisor': ('Investment Advisory\nAgent', '#688EC4')
    }
    
    for role_id, (label, color) in roles.items():
        dot.node(role_id, label=f'Agent: {label}', 
                fillcolor=color, fontcolor='white',
                shape='box3d', width='1.8', height='0.9')

    # Tool nodes with lightest blue shades
    tools = {
        'News Search Tool': ('News Search\nTool', tool_blue),
        'Stock Price Tool': ('Stock Price\nTool', '#85A8E0'),
        'Data Analysis Tool': ('Data Analysis\nTool', '#8FB2E7'),
        'Report Retrieval Tool': ('Report Retrieval\nTool', '#99BCEE'),
        'Report Analysis Tool': ('Report Analysis\nTool', '#A3C6F5'),
        'Text Format Tool': ('Text Format\nTool', '#ADC0FC')
    }
    
    for tool_id, (label, color) in tools.items():
        dot.node(tool_id, label=f'Tool: {label}', 
                fillcolor=color, fontcolor='white',
                shape='cylinder', width='1.7', height='0.8')

    # Forward edges from company to tasks
    for task in tasks:
        dot.edge('Company', task, color=primary_blue, penwidth='2')

    # Task to role connections with feedback
    task_to_role = [
        ('News Search Task', 'News Searcher'),
        ('Price Track Task', 'Price Tracker'),
        ('Report Analysis Task', 'Report Analyst'),
        ('Investment Advice Task', 'Investment Advisor')
    ]
    
    for task, role in task_to_role:
        # Forward arrow
        dot.edge(task, role, color=primary_blue, style='solid', penwidth='1.5',
                label='assigns')
        # Feedback arrow
        dot.edge(role, task, color='#4A6FA5', style='dashed', penwidth='1.5',
                label='feedback', constraint='false')

    # Role to tool connections with feedback
    role_to_tools = [
        ('News Searcher', ['News Search Tool']),
        ('Price Tracker', ['Stock Price Tool', 'Data Analysis Tool']),
        ('Report Analyst', ['Report Retrieval Tool', 'Report Analysis Tool']),
        ('Investment Advisor', ['Text Format Tool'])
    ]
    
    for role, tools_list in role_to_tools:
        for tool in tools_list:
            # Forward arrow
            dot.edge(role, tool, color=primary_blue, style='solid',
                    label='uses')
            # Feedback arrow
            dot.edge(tool, role, color='#4A6FA5', style='dashed',
                    label='returns data', constraint='false')

    return dot

# Create and save the diagram
diagram = create_diagram()
diagram.render('company_tasks_diagram', format='png', cleanup=True)