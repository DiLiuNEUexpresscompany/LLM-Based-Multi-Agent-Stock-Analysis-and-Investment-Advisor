from rich.tree import Tree
from rich.console import Console
from rich.style import Style
from rich.text import Text
from pathlib import Path
import os

def create_project_tree(directory: str = ".") -> Tree:
    """Create a Rich Tree visualization of the project structure."""
    
    # Define styles for different file/folder types
    styles = {
        "folder": Style(color="cyan", bold=True),
        "python": Style(color="green"),
        "markdown": Style(color="yellow"),
        "txt": Style(color="blue"),
        "test": Style(color="magenta"),
        "default": Style(color="white")
    }
    
    def get_style(name: str) -> Style:
        """Determine the style based on file/folder name."""
        if os.path.isdir(name):
            return styles["folder"]
        if name.endswith(".py"):
            if name.startswith("test_"):
                return styles["test"]
            return styles["python"]
        if name.endswith(".md"):
            return styles["markdown"]
        if name.endswith(".txt"):
            return styles["txt"]
        return styles["default"]
    
    def add_to_tree(tree: Tree, path: Path) -> None:
        """Recursively add files and folders to the tree."""
        # Sort directories first, then files
        paths = sorted(
            path.iterdir(),
            key=lambda x: (not x.is_dir(), x.name.lower())
        )
        
        for item in paths:
            # Skip __pycache__ and .git directories
            if item.name in ["__pycache__", ".git", "__init__.py"]:
                continue
                
            # Create styled name with description if available
            name = item.name
            description = ""
            
            # Check for description in file contents
            if item.is_file() and item.suffix == '.py':
                try:
                    with open(item, 'r', encoding='utf-8') as f:
                        first_lines = f.readlines()[:3]
                        for line in first_lines:
                            if '#' in line:
                                description = line.split('#', 1)[1].strip()
                                break
                except Exception:
                    pass
            
            # Create the display text
            if description:
                display = Text(f"{name:<30} ", style=get_style(str(item)))
                display.append(f"# {description}", style=Style(color="grey74", italic=True))
            else:
                display = Text(name, style=get_style(str(item)))
            
            # Add to tree
            if item.is_dir():
                branch = tree.add(display)
                add_to_tree(branch, item)
            else:
                tree.add(display)

    # Create the main tree
    console = Console()
    root = Tree(
        Text("Project Structure", style=Style(color="gold1", bold=True)),
        guide_style=Style(color="grey50")
    )
    
    # Build the tree
    add_to_tree(root, Path(directory))
    
    return root

def main():
    """Main function to display the project tree."""
    console = Console()
    
    # Add a title
    console.print("\n[bold gold1]╔══ Project Structure Visualization ══╗[/]\n")
    
    # Create and display the tree
    tree = create_project_tree()
    console.print(tree)
    
    # Add a legend
    console.print("\n[bold gold1]╚══ File Type Legend ══╝[/]")
    console.print("[cyan bold]■[/] Folders")
    console.print("[green]■[/] Python Files")
    console.print("[magenta]■[/] Test Files")
    console.print("[yellow]■[/] Markdown Files")
    console.print("[blue]■[/] Text Files")

if __name__ == "__main__":
    main()