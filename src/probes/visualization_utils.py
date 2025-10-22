"""
Shared utilities for probe visualization

Helper functions used by both interactive and animated visualizers
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import colorsys


@dataclass
class ColorTheme:
    """Color scheme for visualizations"""
    background: str = "#1a1a1a"
    text: str = "#e0e0e0"
    accent: str = "#00ff00"
    high: str = "#ff0000"
    medium: str = "#ffaa00"
    low: str = "#00ff00"
    inactive: str = "#444444"


def confidence_to_color(confidence: float, theme: ColorTheme = None) -> str:
    """
    Convert confidence value to color string

    Args:
        confidence: Float between 0 and 1
        theme: Color theme to use

    Returns:
        Color string (rich markup or hex)
    """
    if theme is None:
        theme = ColorTheme()

    if confidence < 0.2:
        return "dim blue"
    elif confidence < 0.4:
        return "cyan"
    elif confidence < 0.6:
        return "yellow"
    elif confidence < 0.8:
        return "orange1"
    else:
        return "red"


def confidence_to_rgb(confidence: float) -> Tuple[int, int, int]:
    """
    Convert confidence to RGB tuple

    Args:
        confidence: Float between 0 and 1

    Returns:
        (r, g, b) tuple with values 0-255
    """
    # Gradient from blue (low) to red (high)
    # HSV: 240° (blue) to 0° (red)
    hue = (1.0 - confidence) * 240.0 / 360.0
    rgb_float = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return tuple(int(x * 255) for x in rgb_float)


def get_bar_character(confidence: float, style: str = "blocks") -> str:
    """
    Get character for bar visualization

    Args:
        confidence: Float between 0 and 1
        style: "blocks", "dots", "arrows", "smooth"

    Returns:
        Character to display
    """
    if style == "blocks":
        chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    elif style == "dots":
        chars = [' ', '⋅', '·', '∙', '●', '⬤', '⚫', '⬤', '⬤']
    elif style == "arrows":
        chars = [' ', '→', '⇒', '➔', '➜', '➡', '⮕', '⮕', '⮕']
    elif style == "smooth":
        chars = [' ', '░', '░', '▒', '▒', '▓', '▓', '█', '█']
    else:
        chars = ['█'] * 9

    idx = min(int(confidence * 8), 8)
    return chars[idx]


def format_confidence(confidence: float, width: int = 5) -> str:
    """
    Format confidence as percentage string

    Args:
        confidence: Float between 0 and 1
        width: Total width including %

    Returns:
        Formatted string like " 45.2%"
    """
    return f"{confidence * 100:{width}.1f}%"


def truncate_text(text: str, max_len: int, suffix: str = "...") -> str:
    """
    Truncate text to max length with suffix

    Args:
        text: Input text
        max_len: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def create_sparkline(values: List[float], width: int = 20, height: int = 8) -> str:
    """
    Create ASCII sparkline from values

    Args:
        values: List of floats between 0 and 1
        width: Width of sparkline
        height: Number of height levels

    Returns:
        Sparkline string
    """
    if not values:
        return ' ' * width

    # Resample to width
    if len(values) > width:
        step = len(values) / width
        resampled = [values[int(i * step)] for i in range(width)]
    elif len(values) < width:
        resampled = values + [0.0] * (width - len(values))
    else:
        resampled = values

    # Height levels (Unicode block characters)
    chars = ' ▁▂▃▄▅▆▇█'
    sparkline = ''
    for val in resampled:
        idx = min(int(val * height), height)
        sparkline += chars[idx]

    return sparkline


def create_mini_heatmap(
    data: List[List[float]],
    width: int = 10,
    height: int = 5
) -> List[str]:
    """
    Create mini ASCII heatmap

    Args:
        data: 2D list of floats (rows × cols)
        width: Target width
        height: Target height

    Returns:
        List of strings, one per row
    """
    if not data:
        return [' ' * width for _ in range(height)]

    rows = []
    for row_data in data[:height]:
        row_str = ''
        for i in range(min(len(row_data), width)):
            val = row_data[i]
            char = get_bar_character(val, style="smooth")
            row_str += char
        # Pad if needed
        if len(row_str) < width:
            row_str += ' ' * (width - len(row_str))
        rows.append(row_str)

    # Pad rows if needed
    while len(rows) < height:
        rows.append(' ' * width)

    return rows


def get_category_color(category: str) -> str:
    """
    Get color for probe category

    Args:
        category: Category name

    Returns:
        Rich color string
    """
    category_colors = {
        'metacognitive': 'purple',
        'analytical': 'cyan',
        'emotional': 'yellow',
        'social': 'green',
        'creative': 'deep_pink1',
        'memory': 'blue',
        'planning': 'bright_cyan',
        'default': 'white'
    }

    category_lower = category.lower()
    for key, color in category_colors.items():
        if key in category_lower:
            return color

    return category_colors['default']


def format_timestamp(seconds: float) -> str:
    """
    Format timestamp as MM:SS.mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def get_emoji_for_confidence(confidence: float) -> str:
    """
    Get emoji representing confidence level

    Args:
        confidence: Float between 0 and 1

    Returns:
        Emoji string
    """
    if confidence < 0.1:
        return "💤"
    elif confidence < 0.2:
        return "💨"
    elif confidence < 0.3:
        return "💫"
    elif confidence < 0.5:
        return "⚡"
    elif confidence < 0.7:
        return "🔥"
    elif confidence < 0.9:
        return "💥"
    else:
        return "🌟"


def calculate_stats(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistics for a list of values

    Args:
        values: List of floats

    Returns:
        Dict with mean, max, min, std
    """
    if not values:
        return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}

    import statistics

    return {
        'mean': statistics.mean(values),
        'max': max(values),
        'min': min(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0
    }


def wrap_text(text: str, width: int) -> List[str]:
    """
    Wrap text to specified width

    Args:
        text: Input text
        width: Maximum line width

    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + len(current_line) <= width:
            current_line.append(word)
            current_length += word_len
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_len

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def create_progress_bar(
    current: int,
    total: int,
    width: int = 20,
    filled: str = '█',
    empty: str = '░'
) -> str:
    """
    Create progress bar string

    Args:
        current: Current value
        total: Total value
        width: Width of bar
        filled: Character for filled portion
        empty: Character for empty portion

    Returns:
        Progress bar string
    """
    if total == 0:
        return empty * width

    ratio = current / total
    filled_width = int(ratio * width)
    empty_width = width - filled_width

    return filled * filled_width + empty * empty_width


def get_activation_symbol(confidence: float) -> str:
    """
    Get symbol representing activation level

    Args:
        confidence: Float between 0 and 1

    Returns:
        Symbol character
    """
    if confidence < 0.1:
        return '·'
    elif confidence < 0.3:
        return '○'
    elif confidence < 0.5:
        return '◐'
    elif confidence < 0.7:
        return '◑'
    elif confidence < 0.9:
        return '●'
    else:
        return '⬤'
