import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: root
    font.family: appTheme.uiFontFamily
    modal: true
    focus: true
    width: Math.min(parent ? parent.width - 44 : 720, 720)
    height: Math.min(parent ? parent.height - 48 : 680, 680)
    x: parent ? Math.round((parent.width - width) / 2) : 0
    y: parent ? Math.round((parent.height - height) / 2) : 0
    padding: 0
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property string titleText: ""
    property string subtitleText: ""
    property var detailRows: []
    readonly property color overlayColor: appTheme.bgDeepColor
    readonly property color panelColor: appTheme.toneGraphite
    readonly property color sectionColor: appTheme.bgBaseColor
    readonly property color separatorColor: "#38373C"
    readonly property color textColor: appTheme.textColor
    readonly property color mutedTextColor: appTheme.textMutedColor
    readonly property color accentColor: appTheme.accentColor
    readonly property string dataFontFamily: appTheme.dataFontFamily

    Overlay.modal: Rectangle {
        color: root.overlayColor
        opacity: 0.9
    }

    background: Rectangle {
        radius: 14
        color: root.panelColor
        border.width: 0
        layer.enabled: true
    }

    contentItem: ColumnLayout {
        anchors.fill: parent
        anchors.margins: 22
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                Layout.fillWidth: true
                text: root.titleText
                color: root.textColor
                font.pixelSize: 26
                font.weight: 700
                font.letterSpacing: -0.4
                elide: Text.ElideMiddle
            }

            Label {
                Layout.fillWidth: true
                visible: text.length > 0
                text: root.subtitleText
                color: root.mutedTextColor
                font.family: root.dataFontFamily
                font.pixelSize: 12
                wrapMode: Text.WordWrap
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 10
            color: root.sectionColor
            border.width: 0

            ScrollView {
                id: detailsScroll
                anchors.fill: parent
                anchors.margins: 16
                clip: true
                contentWidth: availableWidth

                Column {
                    width: detailsScroll.availableWidth
                    spacing: 14

                    Repeater {
                        model: root.detailRows

                        delegate: Column {
                            width: parent ? parent.width : 0
                            spacing: 8

                            readonly property bool showSection:
                                index === 0 || root.detailRows[index - 1].section !== modelData.section

                            Label {
                                width: parent.width
                                visible: parent.showSection
                                text: modelData.section
                                color: root.accentColor
                                font.pixelSize: 11
                                font.weight: 700
                                font.letterSpacing: 1.6
                            }

                            RowLayout {
                                width: parent.width
                                spacing: 16

                                Label {
                                    Layout.preferredWidth: Math.min(180, parent.width * 0.34)
                                    Layout.alignment: Qt.AlignTop
                                    text: modelData.label
                                    color: root.mutedTextColor
                                    font.pixelSize: 12
                                    font.weight: 600
                                    wrapMode: Text.WordWrap
                                }

                                Label {
                                    Layout.fillWidth: true
                                    Layout.alignment: Qt.AlignTop
                                    text: modelData.value
                                    color: root.textColor
                                    font.family: root.dataFontFamily
                                    font.pixelSize: modelData.emphasized ? 15 : 13
                                    font.weight: modelData.emphasized ? 600 : 400
                                    wrapMode: Text.WordWrap
                                }
                            }

                            Rectangle {
                                width: parent.width
                                height: 1
                                visible: index < root.detailRows.length - 1
                                color: "#2E2E33"
                                opacity: 0.7
                            }
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true

            Item { Layout.fillWidth: true }

            Button {
                text: qsTr("Close")
                font.family: root.dataFontFamily
                onClicked: root.close()
            }
        }
    }
}
