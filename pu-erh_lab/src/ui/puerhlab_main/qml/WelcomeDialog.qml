import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Effects

Dialog {
    id: dialog
    font.family: appTheme.uiFontFamily

    parent: Overlay.overlay
    modal: true
    focus: visible
    closePolicy: Popup.NoAutoClose
    padding: 0
    width: parent ? parent.width : 0
    height: parent ? parent.height : 0
    x: 0
    y: 0

    property Item blurSource: null
    property var recentProjects: []
    property var languageOptions: []
    property int currentLanguageIndex: 0
    property string serviceMessage: ""
    property string headlineFontFamily: appTheme.headlineFontFamily
    readonly property string dataFontFamily: appTheme.dataFontFamily
    property color primaryAccent: "#6D93B7"
    property color secondaryAccent: "#9FC7D8"
    property color textColor: "#F5F1EA"
    property color mutedTextColor: "#B6B0A7"
    property color panelColor: "#1C1C1D"
    property color panelBorderColor: Qt.rgba(1, 1, 1, 0.08)
    property color overlayColor: Qt.rgba(11 / 255, 12 / 255, 14 / 255, 0.60)
    property color baseColor: "#111214"
    property color exitColor: "#D3D0CB"
    property bool showAllRecent: false
    readonly property int collapsedRecentCount: 4

    signal loadRequested()
    signal createRequested()
    signal exitRequested()
    signal languageRequested(string languageCode)
    signal recentProjectRequested(string projectPath)

    onVisibleChanged: {
        if (visible) {
            showAllRecent = false
        }
    }

    function relativeTimeLabel(lastOpenedMs) {
        const value = Number(lastOpenedMs)
        if (!isFinite(value) || value <= 0) {
            return qsTr("Opened recently")
        }

        const deltaMinutes = Math.max(0, Math.floor((Date.now() - value) / 60000))
        if (deltaMinutes < 1) {
            return qsTr("Opened just now")
        }
        if (deltaMinutes < 60) {
            return qsTr("Opened %n minute(s) ago", "", deltaMinutes)
        }

        const deltaHours = Math.floor(deltaMinutes / 60)
        if (deltaHours < 24) {
            return qsTr("Opened %n hour(s) ago", "", deltaHours)
        }

        const deltaDays = Math.floor(deltaHours / 24)
        if (deltaDays === 1) {
            return qsTr("Opened yesterday")
        }
        if (deltaDays < 7) {
            return qsTr("Opened %n day(s) ago", "", deltaDays)
        }
        if (deltaDays < 14) {
            return qsTr("Opened last week")
        }
        return qsTr("Opened %n day(s) ago", "", deltaDays)
    }

    Overlay.modal: Item {
        anchors.fill: parent

        MultiEffect {
            anchors.fill: parent
            source: dialog.blurSource
            blurEnabled: true
            blur: 0.72
            blurMax: 72
            saturation: -0.24
            brightness: -0.08
        }

        Rectangle {
            anchors.fill: parent
            color: dialog.overlayColor
        }

        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
        }
    }

    background: Item {}

    contentItem: Item {
        implicitWidth: dialog.width
        implicitHeight: dialog.height

        Rectangle {
            id: shell
            anchors.centerIn: parent
            width: Math.min(parent.width - 40, 1120)
            height: Math.min(parent.height - 52, 640)
            radius: 28
            color: Qt.rgba(dialog.panelColor.r, dialog.panelColor.g, dialog.panelColor.b, 0.90)
            border.width: 1
            border.color: dialog.panelBorderColor

            RowLayout {
                anchors.fill: parent
                anchors.margins: 26
                spacing: 24

                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: shell.width * 0.44

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 18

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 0

                            Label {
                                text: qsTr("Pu-erh Lab")
                                color: dialog.primaryAccent
                                font.family: dialog.headlineFontFamily
                                font.pixelSize: 44
                                font.weight: 800
                            }
                        }

                        Item {
                            Layout.preferredHeight: 2
                        }

                        Button {
                            id: loadButton
                            Layout.fillWidth: true
                            Layout.preferredHeight: 76
                            text: qsTr("Load Project")
                            icon.source: "qrc:/panel_icons/import.svg"
                            icon.width: 22
                            icon.height: 22
                            icon.color: dialog.textColor
                            display: AbstractButton.TextBesideIcon
                            font.family: dialog.font.family
                            font.pixelSize: 24
                            font.weight: 700
                            leftPadding: 24
                            rightPadding: 24
                            spacing: 14
                            Material.foreground: dialog.textColor
                            onClicked: dialog.loadRequested()

                            background: Rectangle {
                                radius: 20
                                color: loadButton.down
                                       ? Qt.darker(dialog.primaryAccent, 1.18)
                                       : (loadButton.hovered
                                          ? Qt.lighter(dialog.primaryAccent, 1.06)
                                          : dialog.primaryAccent)
                                border.width: 1
                                border.color: Qt.rgba(dialog.secondaryAccent.r, dialog.secondaryAccent.g, dialog.secondaryAccent.b, 0.16)
                            }

                            contentItem: RowLayout {
                                spacing: loadButton.spacing

                                Item {
                                    Layout.preferredWidth: 30
                                    Layout.preferredHeight: 30

                                    Image {
                                        id: loadIconSource
                                        anchors.centerIn: parent
                                        width: loadButton.icon.width
                                        height: loadButton.icon.height
                                        source: loadButton.icon.source
                                        visible: false
                                        asynchronous: true
                                    }

                                    MultiEffect {
                                        anchors.fill: loadIconSource
                                        source: loadIconSource
                                        colorization: 1.0
                                        colorizationColor: loadButton.icon.color
                                    }
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: loadButton.text
                                    color: dialog.textColor
                                    font: loadButton.font
                                    verticalAlignment: Text.AlignVCenter
                                }

                                Label {
                                    text: "\u2192"
                                    color: Qt.rgba(dialog.textColor.r, dialog.textColor.g, dialog.textColor.b, 0.78)
                                    font.family: dialog.headlineFontFamily
                                    font.pixelSize: 28
                                    font.weight: 400
                                }
                            }
                        }

                        Button {
                            id: createButton
                            Layout.fillWidth: true
                            Layout.preferredHeight: 76
                            text: qsTr("Create Project")
                            display: AbstractButton.TextBesideIcon
                            font.family: dialog.font.family
                            font.pixelSize: 24
                            font.weight: 700
                            leftPadding: 24
                            rightPadding: 24
                            spacing: 14
                            Material.foreground: dialog.textColor
                            onClicked: dialog.createRequested()

                            background: Rectangle {
                                radius: 20
                                color: createButton.down
                                       ? Qt.rgba(dialog.panelColor.r, dialog.panelColor.g, dialog.panelColor.b, 0.98)
                                       : (createButton.hovered
                                          ? Qt.rgba(dialog.panelColor.r, dialog.panelColor.g, dialog.panelColor.b, 0.90)
                                          : Qt.rgba(dialog.panelColor.r, dialog.panelColor.g, dialog.panelColor.b, 0.80))
                                border.width: 1
                                border.color: Qt.rgba(dialog.textColor.r, dialog.textColor.g, dialog.textColor.b, 0.16)
                            }

                            contentItem: RowLayout {
                                spacing: createButton.spacing

                                Rectangle {
                                    Layout.preferredWidth: 30
                                    Layout.preferredHeight: 30
                                    radius: 15
                                    color: Qt.rgba(dialog.secondaryAccent.r, dialog.secondaryAccent.g, dialog.secondaryAccent.b, 0.90)

                                    Label {
                                        anchors.centerIn: parent
                                        text: "+"
                                        color: dialog.baseColor
                                        font.family: dialog.headlineFontFamily
                                        font.pixelSize: 20
                                        font.weight: 800
                                    }
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: createButton.text
                                    color: dialog.textColor
                                    font: createButton.font
                                    verticalAlignment: Text.AlignVCenter
                                }
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: dialog.serviceMessage.length > 0
                            text: dialog.serviceMessage
                            wrapMode: Text.WordWrap
                            color: dialog.mutedTextColor
                            font.family: dialog.font.family
                            font.pixelSize: 13
                            lineHeight: 1.2
                        }

                        Item {
                            Layout.fillHeight: true
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            ComboBox {
                                id: languageCombo
                                Layout.preferredWidth: 168
                                model: dialog.languageOptions
                                textRole: "label"
                                currentIndex: dialog.currentLanguageIndex
                                font.family: dialog.font.family
                                onActivated: function(index) {
                                    const item = model[index]
                                    if (item) {
                                        dialog.languageRequested(item.code)
                                    }
                                }
                                Material.background: Qt.rgba(dialog.panelColor.r, dialog.panelColor.g, dialog.panelColor.b, 0.92)
                                Material.foreground: dialog.textColor
                            }

                            Item {
                                Layout.fillWidth: true
                            }

                            Button {
                                id: exitButton
                                text: qsTr("Exit Application")
                                flat: true
                                font.family: dialog.font.family
                                font.pixelSize: 16
                                font.weight: 600
                                Material.foreground: exitButton.hovered
                                                     ? dialog.textColor
                                                     : dialog.exitColor
                                onClicked: dialog.exitRequested()

                                contentItem: RowLayout {
                                    spacing: 8

                                    Label {
                                        text: "\u21AA"
                                        color: exitButton.Material.foreground
                                        font.pixelSize: 18
                                    }

                                    Label {
                                        text: exitButton.text
                                        color: exitButton.Material.foreground
                                        font: exitButton.font
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.preferredWidth: shell.width * 0.50
                    radius: 24
                    color: Qt.rgba(0, 0, 0, 0.12)
                    border.width: 1
                    border.color: Qt.rgba(1, 1, 1, 0.04)

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 22
                        spacing: 16

                        RowLayout {
                            Layout.fillWidth: true

                            Label {
                                text: qsTr("Recent Projects")
                                color: dialog.textColor
                                font.family: dialog.font.family
                                font.pixelSize: 26
                                font.weight: 800
                            }

                            Item {
                                Layout.fillWidth: true
                            }

                            Button {
                                visible: dialog.recentProjects.length > dialog.collapsedRecentCount
                                flat: true
                                text: dialog.showAllRecent ? qsTr("Collapse") : qsTr("View All")
                                font.family: dialog.font.family
                                font.pixelSize: 14
                                font.weight: 600
                                Material.foreground: dialog.primaryAccent
                                onClicked: dialog.showAllRecent = !dialog.showAllRecent
                            }
                        }

                        Loader {
                            Layout.fillWidth: true
                            active: dialog.recentProjects.length === 0
                            visible: active
                            Layout.fillHeight: active

                            sourceComponent: Item {
                                Column {
                                    anchors.top: parent.top
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    spacing: 8

                                    Label {
                                        text: qsTr("No recent projects yet")
                                        color: dialog.textColor
                                        font.family: dialog.font.family
                                        font.pixelSize: 20
                                        font.weight: 700
                                    }

                                    Label {
                                        text: qsTr("Projects you open or create here will appear in this list.")
                                        color: dialog.mutedTextColor
                                        font.family: dialog.font.family
                                        font.pixelSize: 14
                                        wrapMode: Text.WordWrap
                                    }
                                }
                            }
                        }

                        ListView {
                            id: recentList
                            Layout.fillWidth: true
                            visible: dialog.recentProjects.length > 0
                            Layout.fillHeight: visible
                            Layout.alignment: Qt.AlignTop
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds
                            spacing: 12
                            model: dialog.showAllRecent
                                   ? dialog.recentProjects
                                   : dialog.recentProjects.slice(0, dialog.collapsedRecentCount)

                            delegate: Item {
                                required property string name
                                required property string path
                                required property string folderPath
                                required property double lastOpenedMs

                                width: ListView.view.width
                                height: 82
                                readonly property bool hovered: rowMouse.containsMouse

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 18
                                    color: hovered
                                           ? Qt.rgba(1, 1, 1, 0.055)
                                           : Qt.rgba(0, 0, 0, 0.22)
                                    border.width: 1
                                    border.color: hovered
                                                  ? Qt.rgba(dialog.primaryAccent.r, dialog.primaryAccent.g, dialog.primaryAccent.b, 0.26)
                                                  : Qt.rgba(1, 1, 1, 0.04)

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 18
                                        anchors.rightMargin: 18
                                        spacing: 14

                                        Rectangle {
                                            Layout.preferredWidth: 42
                                            Layout.preferredHeight: 42
                                            radius: 12
                                            color: Qt.rgba(dialog.primaryAccent.r, dialog.primaryAccent.g, dialog.primaryAccent.b, 0.12)

                                            Item {
                                                anchors.centerIn: parent
                                                width: 22
                                                height: 22

                                                Image {
                                                    id: folderIconSource
                                                    anchors.fill: parent
                                                    source: "qrc:/panel_icons/folder-open.svg"
                                                    visible: false
                                                    asynchronous: true
                                                }

                                                MultiEffect {
                                                    anchors.fill: folderIconSource
                                                    source: folderIconSource
                                                    colorization: 1.0
                                                    colorizationColor: dialog.mutedTextColor
                                                }
                                            }
                                        }

                                        ColumnLayout {
                                            Layout.fillWidth: true
                                            spacing: 2

                                            Label {
                                                Layout.fillWidth: true
                                                text: name
                                                elide: Text.ElideRight
                                                color: dialog.textColor
                                                font.family: dialog.dataFontFamily
                                                font.pixelSize: 18
                                                font.weight: 700
                                            }

                                            Label {
                                                Layout.fillWidth: true
                                                text: dialog.relativeTimeLabel(lastOpenedMs)
                                                color: dialog.mutedTextColor
                                                font.family: dialog.dataFontFamily
                                                font.pixelSize: 12
                                                elide: Text.ElideRight
                                            }

                                            Label {
                                                Layout.fillWidth: true
                                                text: folderPath
                                                color: Qt.rgba(dialog.mutedTextColor.r, dialog.mutedTextColor.g, dialog.mutedTextColor.b, 0.74)
                                                font.family: dialog.dataFontFamily
                                                font.pixelSize: 11
                                                elide: Text.ElideMiddle
                                            }
                                        }
                                    }
                                }

                                MouseArea {
                                    id: rowMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: dialog.recentProjectRequested(path)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
