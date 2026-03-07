import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: root
    contentWidth: availableWidth
    readonly property color cardColor: "#242424"
    readonly property color separatorColor: "#363636"
    readonly property color textColor: appTheme.textColor
    readonly property color mutedTextColor: appTheme.textMutedColor
    readonly property color statsMutedTextColor: "#7B7D7C"

    Component.onCompleted: {
        contentItem.interactive = false
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 12

        // ── Total photo count hero card ─────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.margins: 4
            implicitHeight: heroCol.implicitHeight + 24
            radius: 10
            color: root.cardColor

            ColumnLayout {
                id: heroCol
                anchors.fill: parent
                anchors.margins: 16
                spacing: 4

                Label {
                    text: "Photo Library"
                    color: root.mutedTextColor
                    font.pixelSize: 11
                    font.weight: 600
                    font.letterSpacing: 1.2
                }

                Label {
                    text: albumBackend.totalPhotoCount
                    color: root.textColor
                    font.family: appTheme.dataFontFamily
                    font.pixelSize: 42
                    font.weight: 700
                }

                Label {
                    text: albumBackend.filterInfo
                    color: root.statsMutedTextColor
                    font.pixelSize: 11
                }
            }
        }

        // ── Separator ───────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        // ── By Capture Date ─────────────────────────────────────────────
        StatsCard {
            Layout.fillWidth: true
            Layout.margins: 4
            title: "By Capture Date"
            accentColor: "#5B9BD5"
            model: albumBackend.dateStats
            selectedLabel: albumBackend.statsFilterDate
            onBarClicked: function(label) { albumBackend.ToggleStatsFilter("date", label) }
        }

        // ── Separator ───────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        // ── By Camera Model ─────────────────────────────────────────────
        StatsCard {
            Layout.fillWidth: true
            Layout.margins: 4
            title: "By Camera Model"
            accentColor: "#ED7D31"
            model: albumBackend.cameraStats
            selectedLabel: albumBackend.statsFilterCamera
            onBarClicked: function(label) { albumBackend.ToggleStatsFilter("camera", label) }
        }

        // ── Separator ───────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        // ── By Lens ─────────────────────────────────────────────────────
        StatsCard {
            Layout.fillWidth: true
            Layout.margins: 4
            title: "By Lens"
            accentColor: "#70AD47"
            model: albumBackend.lensStats
            selectedLabel: albumBackend.statsFilterLens
            onBarClicked: function(label) { albumBackend.ToggleStatsFilter("lens", label) }
        }

        // ── Bottom spacer ───────────────────────────────────────────────
        Item {
            Layout.fillWidth: true
            Layout.preferredHeight: 8
        }
    }
}
