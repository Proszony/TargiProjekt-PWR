from __future__ import annotations

from PySide6.QtWidgets import QWidget

COLORS: dict[str, str] = {
    "bg-canvas": "#071018",
    "bg-shell": "#0d1721",
    "bg-shell-deep": "#0a131b",
    "bg-surface": "#132230",
    "bg-surface-hover": "#182b3b",
    "bg-surface-active": "#172b3a",
    "bg-surface-selected": "#1a3040",
    "bg-input": "#101923",
    "bg-chip": "#112130",
    "bg-chip-soft": "#12202d",
    "border-strong": "#1f3141",
    "border-accent": "#254255",
    "border-soft": "#23394b",
    "border-muted": "#192734",
    "text-primary": "#f2f7fb",
    "text-secondary": "#dce7f0",
    "text-muted": "#9fb1c0",
    "text-subtle": "#90a5b6",
    "text-faint": "#6d8091",
    "accent-aqua": "#75d3e0",
    "accent-aqua-strong": "#7de3e1",
    "accent-aqua-soft": "#98edeb",
    "accent-blue": "#38bdf8",
    "accent-green": "#22c55e",
    "accent-amber": "#f59e0b",
    "accent-violet": "#a78bfa",
    "accent-rose": "#f43f5e",
    "danger-bg": "#4d1820",
    "danger-border": "#7a2b3a",
    "danger-hover": "#61202a",
}

RADIUS: dict[str, str] = {
    "sm": "8px",
    "md": "10px",
    "lg": "12px",
}

SPACING: dict[str, str] = {
    "sm": "8px",
    "md": "12px",
    "lg": "16px",
    "xl": "18px",
    "xxl": "22px",
}


def chrome_stylesheet() -> str:
    return f"""
    QWidget {{
        background: {COLORS["bg-canvas"]};
        color: {COLORS["text-secondary"]};
    }}
    QDialog, QMainWindow {{
        background: {COLORS["bg-canvas"]};
        color: {COLORS["text-secondary"]};
    }}
    QLabel {{
        background: transparent;
    }}
    QFrame#PanelCard, QGroupBox#PanelCard, QFrame#MetricCard, QFrame#DialogHero {{
        background: {COLORS["bg-shell"]};
        border: 1px solid {COLORS["border-strong"]};
        border-radius: {RADIUS["lg"]};
    }}
    QLabel#SectionTitle {{
        color: {COLORS["text-primary"]};
        font-size: 18px;
        font-weight: 700;
    }}
    QLabel#SectionSubtitle, QLabel#MutedText, QLabel#FieldHint {{
        color: {COLORS["text-muted"]};
        font-size: 12px;
    }}
    QLabel#HintText {{
        color: {COLORS["accent-aqua"]};
        font-size: 12px;
    }}
    QLabel#HeroEyebrow {{
        color: {COLORS["accent-aqua"]};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    QLabel#HeroTitle {{
        color: {COLORS["text-primary"]};
        font-size: 24px;
        font-weight: 700;
    }}
    QLabel#HeroSummary {{
        color: {COLORS["text-muted"]};
        font-size: 13px;
    }}
    QPushButton {{
        background: {COLORS["bg-surface"]};
        border: 1px solid {COLORS["border-soft"]};
        border-radius: {RADIUS["sm"]};
        color: {COLORS["text-primary"]};
        padding: 10px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {COLORS["bg-surface-hover"]};
        border-color: #2d5369;
    }}
    QPushButton:pressed {{
        background: {COLORS["bg-shell-deep"]};
    }}
    QPushButton:disabled {{
        color: {COLORS["text-faint"]};
        background: {COLORS["bg-input"]};
        border-color: {COLORS["border-muted"]};
    }}
    QPushButton[kind="primary"] {{
        background: {COLORS["accent-aqua-strong"]};
        color: #06222b;
        border-color: {COLORS["accent-aqua-strong"]};
    }}
    QPushButton[kind="primary"]:hover {{
        background: {COLORS["accent-aqua-soft"]};
        border-color: {COLORS["accent-aqua-soft"]};
    }}
    QPushButton[kind="danger"] {{
        background: {COLORS["danger-bg"]};
        color: #ffdbe0;
        border-color: {COLORS["danger-border"]};
    }}
    QPushButton[kind="danger"]:hover {{
        background: {COLORS["danger-hover"]};
    }}
    QListWidget, QTableWidget, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
        background: {COLORS["bg-input"]};
        border: 1px solid {COLORS["border-strong"]};
        border-radius: {RADIUS["sm"]};
        color: {COLORS["text-primary"]};
    }}
    QListWidget, QTableWidget {{
        alternate-background-color: {COLORS["bg-shell-deep"]};
    }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
        min-height: 34px;
        padding: 4px 10px;
    }}
    QComboBox QAbstractItemView {{
        background: {COLORS["bg-shell"]};
        color: {COLORS["text-primary"]};
        selection-background-color: {COLORS["bg-surface-selected"]};
        selection-color: {COLORS["text-primary"]};
    }}
    QHeaderView::section {{
        background: {COLORS["bg-shell-deep"]};
        color: {COLORS["text-muted"]};
        border: none;
        border-right: 1px solid {COLORS["border-strong"]};
        padding: 8px;
        font-weight: 600;
    }}
    QTableWidget {{
        gridline-color: {COLORS["border-muted"]};
        selection-background-color: {COLORS["bg-surface-selected"]};
        selection-color: {COLORS["text-primary"]};
    }}
    QDialogButtonBox QPushButton {{
        min-width: 110px;
    }}
    QCheckBox {{
        color: {COLORS["text-secondary"]};
        spacing: 8px;
    }}
    QTabWidget::pane {{
        border: 1px solid {COLORS["border-strong"]};
        background: {COLORS["bg-shell"]};
        border-radius: {RADIUS["md"]};
    }}
    QTabBar::tab {{
        background: {COLORS["bg-shell-deep"]};
        color: {COLORS["text-muted"]};
        padding: 10px 16px;
        border: 1px solid {COLORS["border-strong"]};
        border-bottom: none;
        min-width: 96px;
        border-top-left-radius: {RADIUS["sm"]};
        border-top-right-radius: {RADIUS["sm"]};
    }}
    QTabBar::tab:selected {{
        background: {COLORS["bg-surface-active"]};
        color: {COLORS["text-primary"]};
    }}
    QMessageBox {{
        background: {COLORS["bg-canvas"]};
    }}
    """


def apply_chrome(widget: QWidget) -> None:
    widget.setStyleSheet(chrome_stylesheet())
