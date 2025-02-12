from graphviz import Digraph
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    TASK = "Task"
    AGENT = "Agent"
    TOOL = "Tool"

@dataclass
class NodeConfig:
    label: str
    color: str
    shape: str = "box"
    width: str = "2"
    height: str = "1"
    style: str = "filled"  # Removed shadowed for cleaner look
    fontcolor: str = "white"
    fontname: str = "Arial"
    margin: str = "0.3"

class ColorScheme:
    # Modern, cohesive color palette
    PRIMARY = "#2D3748"
    TASK = "#4C51BF"
    AGENT = "#667EEA"
    TOOL = "#7F9CF5"
    BACKGROUND = "#FFFFFF"
    SUMMARY = "#ED64A6"
    
    # Sophisticated gradient variations
    TASK_VARIANTS = ["#4C51BF", "#5A67D8", "#6875F5"]
    AGENT_VARIANTS = ["#667EEA", "#7886D7", "#849BE5"]
    TOOL_VARIANTS = ["#7F9CF5", "#8DA2FB", "#9BB6FE"]
    
    # Edge colors
    EDGE_PRIMARY = "#718096"
    EDGE_FEEDBACK = "#000000"
    EDGE_INFO = "#ED64A6"

class AdvisorySystemConfig:
    def __init__(self):
        self.colors = ColorScheme()
        self.nodes = self._initialize_nodes()
        self.relationships = self._initialize_relationships()

    def _initialize_nodes(self) -> Dict[NodeType, Dict[str, NodeConfig]]:
        return {
            NodeType.TASK: {
                'News Search Task': NodeConfig(
                    label="News Search\nSubroutine",
                    color=self.colors.TASK_VARIANTS[0],
                    shape="rect",
                    fontname="Arial Bold"
                ),
                'Market Track Task': NodeConfig(
                    label="Market Tracking\nSubroutine",
                    color=self.colors.TASK_VARIANTS[1],
                    shape="rect",
                    fontname="Arial Bold"
                ),
                'Report Analysis Task': NodeConfig(
                    label="Report Analysis\nSubroutine",
                    color=self.colors.TASK_VARIANTS[2],
                    shape="rect",
                    fontname="Arial Bold"
                ),
                'Advisory Task': NodeConfig(
                    label="Advisory\nSubroutine",
                    color=self.colors.SUMMARY,
                    shape="rect",
                    fontname="Arial Bold"
                )
            },
            NodeType.AGENT: {
                'News Searcher': NodeConfig(
                    label="News Search\nAgent",
                    color=self.colors.AGENT_VARIANTS[0],
                    shape="box3d",
                    width="1.8",
                    height="0.9",
                    fontname="Arial"
                ),
                'Market Tracker': NodeConfig(
                    label="Market Tracking\nAgent",
                    color=self.colors.AGENT_VARIANTS[1],
                    shape="box3d",
                    width="1.8",
                    height="0.9",
                    fontname="Arial"
                ),
                'Report Analyst': NodeConfig(
                    label="Report Analysis\nAgent",
                    color=self.colors.AGENT_VARIANTS[2],
                    shape="box3d",
                    width="1.8",
                    height="0.9",
                    fontname="Arial"
                ),
                'Advisor': NodeConfig(
                    label="Advisory\nAgent",
                    color=self.colors.SUMMARY,
                    shape="box3d",
                    width="1.8",
                    height="0.9",
                    fontname="Arial Bold"
                )
            },
            NodeType.TOOL: {
                'News Search Tool': NodeConfig(
                    label="News Search\nTool",
                    color=self.colors.TOOL_VARIANTS[0],
                    shape="cylinder",
                    width="1.7",
                    height="0.8",
                    fontname="Arial"
                ),
                'Market Data Tool': NodeConfig(
                    label="Market Data Tool, \nStock Price Tool",
                    color=self.colors.TOOL_VARIANTS[1],
                    shape="cylinder",
                    width="1.7",
                    height="0.8",
                    fontname="Arial"
                ),
                'Report Tool': NodeConfig(
                    label="Report Retrieval Tool, \nReport Analysis Tool",
                    color=self.colors.TOOL_VARIANTS[2],
                    shape="cylinder",
                    width="1.7",
                    height="0.8",
                    fontname="Arial"
                ),
                'Summary Tool': NodeConfig(
                    label="Information\nSynthesis Tool",
                    color=self.colors.SUMMARY,
                    shape="cylinder",
                    width="1.7",
                    height="0.8",
                    fontname="Arial Bold"
                )
            }
        }

    def _initialize_relationships(self) -> Dict[str, List[Tuple[str, str, dict]]]:
        return {
            'task_agent': [
                ('News Search Task', 'News Searcher', {'color': self.colors.EDGE_PRIMARY, 'label': 'assigns', 'fontname': 'Arial'}),
                ('Market Track Task', 'Market Tracker', {'color': self.colors.EDGE_PRIMARY, 'label': 'assigns', 'fontname': 'Arial'}),
                ('Report Analysis Task', 'Report Analyst', {'color': self.colors.EDGE_PRIMARY, 'label': 'assigns', 'fontname': 'Arial'}),
                ('Advisory Task', 'Advisor', {'color': self.colors.EDGE_PRIMARY, 'label': 'assigns', 'fontname': 'Arial'})
            ],
            'agent_tool': [
                ('News Searcher', 'News Search Tool', {'color': self.colors.EDGE_PRIMARY, 'label': 'uses', 'fontname': 'Arial'}),
                ('Market Tracker', 'Market Data Tool', {'color': self.colors.EDGE_PRIMARY, 'label': 'uses', 'fontname': 'Arial'}),
                ('Report Analyst', 'Report Tool', {'color': self.colors.EDGE_PRIMARY, 'label': 'uses', 'fontname': 'Arial'}),
                ('Advisor', 'Summary Tool', {'color': self.colors.EDGE_PRIMARY, 'label': 'uses', 'fontname': 'Arial'})
            ],
            'info_flow': [
                ('News Search Task', 'Advisor', {'color': self.colors.EDGE_INFO, 'style': 'bold', 'label': '', 'penwidth': '2', 'fontname': 'Arial'}),
                ('Market Track Task', 'Advisor', {'color': self.colors.EDGE_INFO, 'style': 'bold', 'label': '', 'penwidth': '2', 'fontname': 'Arial'}),
                ('Report Analysis Task', 'Advisor', {'color': self.colors.EDGE_INFO, 'style': 'bold', 'label': '', 'penwidth': '2', 'fontname': 'Arial'})
            ],
            'feedback': [
                ('News Searcher', 'News Search Task', {'color': self.colors.EDGE_FEEDBACK, 'label': 'feedback', 'dir': 'back','penwidth': '2', 'constraint': 'false', 'fontname': 'Arial'}),
                ('Market Tracker', 'Market Track Task', {'color': self.colors.EDGE_FEEDBACK, 'label': 'feedback', 'dir': 'back', 'penwidth': '2', 'constraint': 'false', 'fontname': 'Arial'}),
                ('Report Analyst', 'Report Analysis Task', {'color': self.colors.EDGE_FEEDBACK, 'label': 'feedback', 'dir': 'back', 'penwidth': '2', 'constraint': 'false', 'fontname': 'Arial'}),
                ('Advisor', 'Advisory Task', {'color': self.colors.EDGE_FEEDBACK, 'label': 'feedback', 'dir': 'back', 'penwidth': '2', 'constraint': 'false', 'fontname': 'Arial'})
            ]
        }

class AdvisoryDiagramGenerator:
    def __init__(self, config: Optional[AdvisorySystemConfig] = None):
        self.config = config or AdvisorySystemConfig()

    def _initialize_diagram(self) -> Digraph:
        dot = Digraph(format='png')
        dot.attr(
            rankdir='TD',
            size='20,30',
            bgcolor=self.config.colors.BACKGROUND,
            splines='curved',  # Changed to curved for more organic feel
            compound='true',
            pad='0.5'
        )
        
        # Enhanced default node attributes
        dot.attr('node',
                style='filled',
                fontname='Arial',
                fontsize='12',
                penwidth='1.5',
                margin='0.3')
        
        # Enhanced default edge attributes
        dot.attr('edge',
                fontname='Arial',
                fontsize='10',
                penwidth='1.2',
                arrowsize='0.7')
        
        return dot

    def _add_nodes(self, dot: Digraph) -> None:
        for node_type, nodes in self.config.nodes.items():
            for node_id, config in nodes.items():
                dot.node(
                    node_id,
                    label=f"{config.label}",  # Removed node type from label for cleaner look
                    fillcolor=config.color,
                    shape=config.shape,
                    width=config.width,
                    height=config.height,
                    style=config.style,
                    fontcolor=config.fontcolor,
                    fontname=config.fontname,
                    margin=config.margin
                )

    def _add_relationships(self, dot: Digraph) -> None:
        for relationship_type, edges in self.config.relationships.items():
            for source, target, attrs in edges:
                dot.edge(source, target, **attrs)

    def _add_advisory_cluster(self, dot: Digraph) -> None:
        with dot.subgraph(name='cluster_advisory') as c:
            c.attr(
                label='Advisory System',
                style='rounded',
                color=self.config.colors.SUMMARY,
                fillcolor=f"{self.config.colors.SUMMARY}10",
                fontcolor=self.config.colors.SUMMARY,
                fontname='Arial Bold',
                penwidth='2',
                margin='20'
            )
            c.node('Advisory Task')
            c.node('Advisor')
            c.node('Summary Tool')

    def create_diagram(self) -> Digraph:
        dot = self._initialize_diagram()
        self._add_nodes(dot)
        self._add_relationships(dot)
        self._add_advisory_cluster(dot)
        return dot

def generate_diagram(output_file: str = 'advisory_system_diagram') -> None:
    generator = AdvisoryDiagramGenerator()
    diagram = generator.create_diagram()
    diagram.render(output_file, format='png', cleanup=True)

if __name__ == '__main__':
    generate_diagram()