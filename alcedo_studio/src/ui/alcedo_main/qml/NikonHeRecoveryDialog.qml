import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

Popup {
    id: root
    font.family: appTheme.uiFontFamily
    modal: true
    focus: true
    visible: recoveryActive
    closePolicy: Popup.NoAutoClose
    anchors.centerIn: parent
    width: Math.min(parent ? parent.width - 52 : 860, 860)
    height: Math.min(parent ? parent.height - 56 : 760, 760)
    padding: 0

    property bool recoveryActive: false
    property bool recoveryBusy: false
    property bool showImportProgress: false
    property int importCompleted: 0
    property int importTotal: 0
    property int importFailed: 0
    property string recoveryPhase: ""
    property string recoveryStatus: ""
    property string converterPath: ""
    property var unsupportedFiles: []
    property Item backgroundSource: null

    signal browseRequested()
    signal convertRequested()
    signal exitRequested()

    readonly property color panelColor: appTheme.bgDeepColor
    readonly property color sectionColor: Qt.rgba(0.12, 0.12, 0.12, 0.92)
    readonly property color overlayTint: Qt.rgba(0.03, 0.03, 0.04, 0.76)
    readonly property color warmAccent: appTheme.toneGold
    readonly property color warmAccentSoft: Qt.rgba(warmAccent.r, warmAccent.g, warmAccent.b, 0.16)
    readonly property color separatorColor: Qt.rgba(1, 1, 1, 0.08)
    readonly property color mutedText: appTheme.textMutedColor
    readonly property color strongText: appTheme.textColor
    readonly property bool showBusyIndicator: root.recoveryBusy && !root.showImportProgress
    readonly property bool macOsUsesFixedConverterPath: Qt.platform.os === "osx"
    readonly property bool converterDetected: root.converterPath.length > 0

    Overlay.modal: Item {
        anchors.fill: parent

        MultiEffect {
            visible: root.backgroundSource !== null
            anchors.fill: parent
            source: root.backgroundSource
            blurEnabled: true
            blur: 0.72
            blurMax: 72
            saturation: -0.26
        }

        Rectangle {
            anchors.fill: parent
            gradient: Gradient {
                GradientStop { position: 0.0; color: Qt.rgba(0.06, 0.06, 0.06, 0.84) }
                GradientStop { position: 0.65; color: root.overlayTint }
                GradientStop { position: 1.0; color: Qt.rgba(0.09, 0.08, 0.05, 0.84) }
            }
        }

        MouseArea { anchors.fill: parent; hoverEnabled: true }
    }

    background: Rectangle {
        radius: 18
        color: root.panelColor
        border.width: 1
        border.color: Qt.rgba(root.warmAccent.r, root.warmAccent.g, root.warmAccent.b, 0.18)
    }

    contentItem: ScrollView {
        id: scrollView
        anchors.fill: parent
        contentWidth: width - 56
        padding: 28
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        ColumnLayout {
            width: scrollView.width - 56
            spacing: 22

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 10

                Label {
                    text: root.recoveryPhase.length > 0 ? root.recoveryPhase.toUpperCase() : qsTr("RECOVERY")
                    color: root.warmAccent
                    font.pixelSize: 11
                    font.weight: 800
                    font.letterSpacing: 1.8
                }

                Label {
                    Layout.fillWidth: true
                    text: qsTr("Nikon HE / HE* RAW Needs Conversion")
                    color: root.strongText
                    font.pixelSize: 30
                    font.weight: 800
                    wrapMode: Text.WordWrap
                }

                Label {
                    Layout.fillWidth: true
                    Layout.maximumWidth: Math.max(420, root.width * 0.72)
                    text: qsTr("Alcedo Studio cannot decode these Nikon files with the built-in RAW pipeline. Convert them to DNG, then the import can continue.")
                    color: root.mutedText
                    wrapMode: Text.WordWrap
                    font.pixelSize: 14
                    lineHeight: 1.2
                    Layout.minimumWidth: 100
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: statusColumn.implicitHeight + 32
                radius: 14
                color: root.warmAccentSoft
                border.width: 1
                border.color: Qt.rgba(root.warmAccent.r, root.warmAccent.g, root.warmAccent.b, 0.24)
                clip: true

                ColumnLayout {
                    id: statusColumn
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 12

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        BusyIndicator {
                            Layout.alignment: Qt.AlignTop
                            running: root.showBusyIndicator
                            visible: root.showBusyIndicator
                            implicitWidth: 28
                            implicitHeight: 28
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                Layout.fillWidth: true
                                text: root.recoveryStatus.length > 0
                                      ? root.recoveryStatus
                                      : qsTr("Preparing recovery workflow...")
                                color: root.strongText
                                wrapMode: Text.WordWrap
                                font.pixelSize: 15
                                font.weight: 700
                                lineHeight: 1.18
                                Layout.minimumWidth: 50
                            }

                            Label {
                                visible: root.showBusyIndicator
                                text: qsTr("This step is still running. The dialog will update automatically when conversion finishes.")
                                color: root.mutedText
                                wrapMode: Text.WordWrap
                                font.pixelSize: 12
                                Layout.minimumWidth: 50
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: root.showImportProgress
                        spacing: 6

                        ProgressBar {
                            Layout.fillWidth: true
                            from: 0
                            to: Math.max(1, root.importTotal)
                            value: root.importCompleted + root.importFailed
                        }

                        Label {
                            text: qsTr("%1 / %2 converted files reimported · %3 failed")
                                  .arg(root.importCompleted)
                                  .arg(root.importTotal)
                                  .arg(root.importFailed)
                            color: root.mutedText
                            font.family: appTheme.dataFontFamily
                            font.pixelSize: 12
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignTop
                spacing: 16

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredWidth: Math.max(350, root.width * 0.6)
                    Layout.preferredHeight: Math.max(320, Math.min(460, root.height * 0.46))
                    radius: 16
                    color: root.sectionColor
                    border.width: 1
                    border.color: root.separatorColor

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 18
                        spacing: 12

                        Label {
                            text: qsTr("Affected Files")
                            color: root.strongText
                            font.pixelSize: 17
                            font.weight: 700
                        }

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("%1 file(s) will be removed from the project and replaced by converted DNG files when available.")
                                  .arg(root.unsupportedFiles.length)
                            color: root.mutedText
                            wrapMode: Text.WordWrap
                            font.pixelSize: 12
                            Layout.minimumWidth: 50
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 12
                            color: Qt.rgba(0, 0, 0, 0.18)
                            border.width: 1
                            border.color: root.separatorColor

                            ScrollView {
                                anchors.fill: parent
                                anchors.margins: 10
                                clip: true

                                Column {
                                    width: parent.width
                                    spacing: 8

                                    Repeater {
                                        model: root.unsupportedFiles

                                        delegate: Rectangle {
                                            width: parent.width
                                            radius: 10
                                            color: Qt.rgba(1, 1, 1, 0.04)
                                            border.width: 1
                                            border.color: Qt.rgba(root.warmAccent.r, root.warmAccent.g, root.warmAccent.b, 0.12)
                                            height: fileColumn.implicitHeight + 18

                                            ColumnLayout {
                                                id: fileColumn
                                                anchors.fill: parent
                                                anchors.margins: 10
                                                spacing: 4

                                                Label {
                                                    Layout.fillWidth: true
                                                    text: modelData.fileName
                                                    color: root.strongText
                                                    font.pixelSize: 13
                                                    font.weight: 700
                                                    elide: Text.ElideMiddle
                                                }

                                                Label {
                                                    Layout.fillWidth: true
                                                    text: modelData.sourcePath
                                                    color: root.mutedText
                                                    font.family: appTheme.dataFontFamily
                                                    font.pixelSize: 11
                                                    elide: Text.ElideMiddle
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredWidth: Math.max(260, root.width * 0.32)
                    Layout.preferredHeight: Math.max(320, Math.min(460, root.height * 0.46))
                    Layout.alignment: Qt.AlignTop
                    radius: 16
                    color: root.sectionColor
                    border.width: 1
                    border.color: root.separatorColor

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 18
                        spacing: 14

                        Label {
                            text: qsTr("Converter")
                            color: root.strongText
                            font.pixelSize: 17
                            font.weight: 700
                        }

                        Label {
                            Layout.fillWidth: true
                            text: root.macOsUsesFixedConverterPath
                                  ? qsTr("On macOS, Alcedo Studio uses the system Adobe DNG Converter installation at /Applications/Adobe DNG Converter.app.")
                                  : qsTr("Adobe DNG Converter runs in the source folders and creates side-by-side DNG files. Original NEF files on disk will be kept.")
                            color: root.mutedText
                            wrapMode: Text.WordWrap
                            font.pixelSize: 12
                            Layout.minimumWidth: 50
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            radius: 12
                            color: Qt.rgba(0, 0, 0, 0.18)
                            border.width: 1
                            border.color: root.separatorColor
                            height: converterColumn.implicitHeight + 18

                            ColumnLayout {
                                id: converterColumn
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 8

                                Label {
                                    Layout.fillWidth: true
                                    text: root.macOsUsesFixedConverterPath
                                          ? qsTr("Detected Converter Path")
                                          : qsTr("Executable Path")
                                    color: root.warmAccent
                                    font.pixelSize: 11
                                    font.weight: 700
                                    font.letterSpacing: 1.4
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: root.converterPath.length > 0
                                          ? root.converterPath
                                          : (root.macOsUsesFixedConverterPath
                                             ? qsTr("Adobe DNG Converter is not installed at /Applications/Adobe DNG Converter.app.")
                                             : qsTr("Not selected in this session."))
                                    color: root.strongText
                                    font.family: appTheme.dataFontFamily
                                    font.pixelSize: 12
                                    wrapMode: Text.WrapAnywhere
                                    Layout.minimumWidth: 50
                                }

                                Button {
                                    text: qsTr("Browse Converter...")
                                    visible: !root.macOsUsesFixedConverterPath
                                    enabled: !root.recoveryBusy
                                    onClicked: root.browseRequested()
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("Exit removes the unsupported Nikon entries from the project only. Nothing in the source folders will be deleted.")
                            color: root.mutedText
                            wrapMode: Text.WordWrap
                            font.pixelSize: 12
                            Layout.minimumWidth: 50
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 12

                Item { Layout.fillWidth: true }

                Button {
                    text: qsTr("Exit And Remove From Project")
                    enabled: !root.recoveryBusy
                    onClicked: root.exitRequested()
                }

                Button {
                    text: root.macOsUsesFixedConverterPath
                          ? qsTr("Convert To DNG And Continue")
                          : (root.converterPath.length > 0
                             ? qsTr("Convert To DNG And Continue")
                             : qsTr("Choose Converter And Continue"))
                    enabled: !root.recoveryBusy
                             && (!root.macOsUsesFixedConverterPath || root.converterDetected)
                    Material.background: appTheme.toneGold
                    Material.foreground: "#16130C"
                    onClicked: root.convertRequested()
                }
            }
        }
    }
}
