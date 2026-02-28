import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/*  Reusable dashboard card for displaying GROUP BY statistics.
 *
 *  Expected model: QVariantList of QVariantMap {label: string, count: int}
 *  Properties:
 *      title       – section heading
 *      accentColor – bar fill colour
 *      model       – the QVariantList
 */
Rectangle {
    id: card
    radius: 8
    color: "#1F1F1F"
    border.width: 0
    implicitHeight: cardCol.implicitHeight + 24

    // ── Public interface ────────────────────────────────────────────
    property string title: ""
    property color accentColor: "#5B9BD5"
    property var model: []
    property bool expanded: true
    readonly property int previewLimit: 10

    // Derived
    readonly property int totalItems: model ? model.length : 0
    readonly property bool hasOverflow: totalItems > previewLimit
    property bool showAll: false
    readonly property int visibleCount: showAll ? totalItems
                                                : Math.min(totalItems, previewLimit)

    // Compute maximum count for proportional bars
    function maxCount() {
        let m = 1;
        if (!model) return m;
        const n = Math.min(model.length, visibleCount);
        for (let i = 0; i < n; ++i) {
            const c = Number(model[i].count);
            if (c > m) m = c;
        }
        return m;
    }

    ColumnLayout {
        id: cardCol
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        // ── Header row ──────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true

            Label {
                text: card.title
                color: "#E3DFDB"
                font.pixelSize: 13
                font.weight: 600
            }

            Label {
                text: "(" + card.totalItems + ")"
                color: "#7B7D7C"
                font.pixelSize: 11
            }

            Item { Layout.fillWidth: true }

            Label {
                text: card.expanded ? "▲" : "▼"
                color: "#7B7D7C"
                font.pixelSize: 11
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: card.expanded = !card.expanded
                }
            }
        }

        // ── Bar list ────────────────────────────────────────────────
        ColumnLayout {
            visible: card.expanded
            Layout.fillWidth: true
            spacing: 3

            Repeater {
                model: card.visibleCount

                delegate: Item {
                    required property int index
                    Layout.fillWidth: true
                    implicitHeight: 24

                    readonly property var entry: card.model[index]
                    readonly property string entryLabel: entry ? String(entry.label) : ""
                    readonly property int entryCount: entry ? Number(entry.count) : 0
                    readonly property real fraction: entryCount / card.maxCount()

                    // Background bar (proportional width)
                    Rectangle {
                        anchors.left: parent.left
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width * fraction
                        height: 20
                        radius: 3
                        color: card.accentColor
                        opacity: 0.25
                    }

                    // Labels (overlay)
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 6
                        anchors.rightMargin: 6

                        Label {
                            Layout.fillWidth: true
                            text: entryLabel
                            color: "#E3DFDB"
                            font.pixelSize: 11
                            elide: Text.ElideRight
                        }

                        Label {
                            text: entryCount
                            color: "#AAAAAA"
                            font.pixelSize: 11
                            font.weight: 600
                        }
                    }
                }
            }

            // ── "Show more / less" toggle ───────────────────────────
            Label {
                visible: card.hasOverflow && card.expanded
                text: card.showAll ? "Show less ▲" : "Show all " + card.totalItems + " ▼"
                color: card.accentColor
                font.pixelSize: 11
                font.underline: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 4

                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: card.showAll = !card.showAll
                }
            }

            // ── Empty state ─────────────────────────────────────────
            Label {
                visible: card.totalItems === 0 && card.expanded
                text: "No data available"
                color: "#555555"
                font.pixelSize: 11
                font.italic: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 8
            }
        }
    }
}
