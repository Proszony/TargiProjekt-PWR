from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)


SAMPLE_SOURCES = [
    {
        "title": "Decorated hall overhead",
        "angle": "High angle, wide floor visibility",
        "use_case": "Good for multi-person tracking and zone testing.",
        "notes": "Pexels clip with overhead movement and good separation between people.",
        "url": "https://www.pexels.com/video/aerial-view-of-people-walking-in-decorated-hall-36164228/",
    },
    {
        "title": "Large indoor atrium",
        "angle": "Wide indoor hall, elevated side view",
        "use_case": "Good for booth occupancy and aisle flow.",
        "notes": "Large open space with several people moving through distinct regions.",
        "url": "https://www.pexels.com/video/a-large-indoor-space-with-people-walking-around-20660922/",
    },
    {
        "title": "Shopping mall atrium",
        "angle": "Elevated atrium overview",
        "use_case": "Good for return-rate and multi-zone transitions.",
        "notes": "Useful when you want repeated entries/exits across neighboring sectors.",
        "url": "https://www.pexels.com/video/the-atrium-of-a-shopping-mall-with-people-walking-around-19765671/",
    },
]


class SampleCatalogDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sample test videos")
        self.resize(900, 520)
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        content = QHBoxLayout()

        self.list_widget = QListWidget()
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        self.details.setReadOnly(True)

        content.addWidget(self.list_widget, 1)
        content.addWidget(self.details, 2)

        helper = QLabel(
            "Download one of the clips locally, then switch Source type to Local MP4 and point the app at the file."
        )
        helper.setWordWrap(True)

        self.open_button = QPushButton("Open link")
        self.copy_button = QPushButton("Copy URL")
        button_row = QHBoxLayout()
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.copy_button)
        button_row.addStretch(1)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(self.reject)

        root.addLayout(content, 1)
        root.addWidget(helper)
        root.addLayout(button_row)
        root.addWidget(close_box)

        self.list_widget.currentRowChanged.connect(self._render_current)
        self.open_button.clicked.connect(self._open_current)
        self.copy_button.clicked.connect(self._copy_current)

    def _populate(self) -> None:
        for source in SAMPLE_SOURCES:
            item = QListWidgetItem(source["title"])
            item.setData(256, source)
            self.list_widget.addItem(item)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _current_source(self) -> dict[str, str] | None:
        item = self.list_widget.currentItem()
        if item is None:
            return None
        data = item.data(256)
        return data if isinstance(data, dict) else None

    def _render_current(self) -> None:
        source = self._current_source()
        if source is None:
            self.details.clear()
            return
        self.details.setHtml(
            f"""
            <h3>{source['title']}</h3>
            <p><b>Angle:</b> {source['angle']}</p>
            <p><b>Use case:</b> {source['use_case']}</p>
            <p><b>Notes:</b> {source['notes']}</p>
            <p><b>URL:</b><br>{source['url']}</p>
            """
        )

    def _open_current(self) -> None:
        source = self._current_source()
        if source is None:
            return
        QDesktopServices.openUrl(QUrl(source["url"]))

    def _copy_current(self) -> None:
        source = self._current_source()
        if source is None:
            return
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(source["url"])
