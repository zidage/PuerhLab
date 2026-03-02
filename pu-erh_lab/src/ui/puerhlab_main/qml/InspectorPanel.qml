import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: root
    contentWidth: availableWidth

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
            color: "#242424"

            ColumnLayout {
                id: heroCol
                anchors.fill: parent
                anchors.margins: 16
                spacing: 4

                Label {
                    text: "Photo Library"
                    color: "#888888"
                    font.pixelSize: 11
                    font.weight: 600
                    font.letterSpacing: 1.2
                }

                Label {
                    text: albumBackend.totalPhotoCount
                    color: "#E3DFDB"
                    font.pixelSize: 42
                    font.weight: 700
                }

                Label {
                    text: albumBackend.filterInfo
                    color: "#7B7D7C"
                    font.pixelSize: 11
                }
            }
        }

        // ── Separator ───────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: "#363636"
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
            color: "#363636"
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
            color: "#363636"
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
