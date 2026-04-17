import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ScrollView {
    id: root
    contentWidth: availableWidth
    readonly property color textColor: appTheme.textColor
    readonly property color mutedTextColor: appTheme.textMutedColor

    Component.onCompleted: {
        contentItem.interactive = false
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: 0

        // Library Overview hero
        Item {
            Layout.fillWidth: true
            Layout.topMargin: 24
            Layout.leftMargin: 16
            Layout.rightMargin: 16
            Layout.bottomMargin: 4
            implicitHeight: heroCol.implicitHeight

            ColumnLayout {
                id: heroCol
                anchors.left: parent.left
                anchors.right: parent.right
                spacing: 14

                Label {
                    text: qsTr("LIBRARY OVERVIEW")
                    color: root.mutedTextColor
                    font.pixelSize: 10
                    font.weight: 700
                    font.letterSpacing: 1.8
                }

                RowLayout {
                    Layout.fillWidth: true

                    Label {
                        text: qsTr("Total Photos")
                        color: root.mutedTextColor
                        font.family: appTheme.uiFontFamily
                        font.pixelSize: 13
                        font.weight: 400
                        Layout.alignment: Qt.AlignVCenter
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: albumBackend.totalPhotoCount
                        color: root.textColor
                        font.family: appTheme.headlineFontFamily
                        font.pixelSize: 34
                        font.weight: 300
                        Layout.alignment: Qt.AlignVCenter
                    }
                }

                Label {
                    visible: albumBackend.filterInfo !== ""
                    text: albumBackend.filterInfo
                    color: root.mutedTextColor
                    font.family: appTheme.uiFontFamily
                    font.pixelSize: 11
                    font.weight: 400
                    Layout.topMargin: -6
                }
            }
        }

        // Stats sections
        ColumnLayout {
            Layout.fillWidth: true
            Layout.topMargin: 28
            Layout.leftMargin: 16
            Layout.rightMargin: 16
            Layout.bottomMargin: 20
            spacing: 24

            StatsCard {
                Layout.fillWidth: true
                title: qsTr("By Capture Date")
                accentColor: "#5B9BD5"
                model: albumBackend.dateStats
                selectedLabel: albumBackend.statsFilterDate
                displayMode: "grouped"
                onBarClicked: function(label) { albumBackend.ToggleStatsFilter("date", label) }
            }

            StatsCard {
                Layout.fillWidth: true
                title: qsTr("By Camera Model")
                accentColor: "#ED7D31"
                model: albumBackend.cameraStats
                selectedLabel: albumBackend.statsFilterCamera
                displayMode: "chips"
                onBarClicked: function(label) { albumBackend.ToggleStatsFilter("camera", label) }
            }

            StatsCard {
                Layout.fillWidth: true
                title: qsTr("By Lens")
                accentColor: "#70AD47"
                model: albumBackend.lensStats
                selectedLabel: albumBackend.statsFilterLens
                displayMode: "dots"
                onBarClicked: function(label) { albumBackend.ToggleStatsFilter("lens", label) }
            }
        }
    }
}
