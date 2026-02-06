import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: root
    width: 1460
    height: 900
    visible: true
    title: "pu-erh_lab - Album + Editor (QML POC)"

    Material.theme: Material.Dark
    Material.accent: "#5AA2FF"
    color: "#0E131A"

    property bool settingsPage: false
    property bool inspectorVisible: true
    property bool gridMode: true
    property bool drawerOpen: true

    function idxByValue(options, v) {
        for (let i = 0; i < options.length; ++i) {
            if (Number(options[i].value) === Number(v)) {
                return i
            }
        }
        return 0
    }

    function applyFilter() {
        albumBackend.applyFilters(quickSearch.text, joinCombo.currentValue)
    }

    FileDialog {
        id: importDialog
        title: "Select Images"
        fileMode: FileDialog.OpenFiles
        nameFilters: [
            "Images (*.dng *.nef *.cr2 *.cr3 *.arw *.rw2 *.raf *.tif *.tiff *.jpg *.jpeg *.png)",
            "All Files (*)"
        ]
        onAccepted: {
            const files = []
            for (let i = 0; i < selectedFiles.length; ++i) {
                files.push(selectedFiles[i].toString())
            }
            albumBackend.startImport(files)
        }
    }

    FolderDialog {
        id: exportDialog
        title: "Select Export Folder"
        onAccepted: albumBackend.startExport(selectedFolder.toString())
    }

    Rectangle {
        anchors.fill: parent
        z: -1
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#0E131A" }
            GradientStop { position: 1.0; color: "#101A26" }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 14
        spacing: 10

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 58
            radius: 14
            color: "#172231"
            border.color: "#32465F"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                Label { text: "PuerhLab"; font.pixelSize: 19; font.weight: 700; color: "#F2F7FF" }
                Button { text: "Library"; checkable: true; checked: !settingsPage; onClicked: settingsPage = false }
                Button { text: "Settings"; checkable: true; checked: settingsPage; onClicked: settingsPage = true }
                TextField { Layout.fillWidth: true; placeholderText: "Search photos" }
                Button { text: "Import"; onClicked: importDialog.open() }
                Button { text: "Export"; enabled: albumBackend.shownCount > 0; onClicked: exportDialog.open() }
                Button { text: "Inspector"; checkable: true; checked: inspectorVisible; onToggled: inspectorVisible = checked }
            }
        }

        Rectangle {
            visible: albumBackend.serviceMessage.length > 0
            Layout.fillWidth: true
            Layout.preferredHeight: 34
            radius: 10
            color: albumBackend.serviceReady ? "#133125" : "#432929"
            border.color: albumBackend.serviceReady ? "#2A715A" : "#7D4A4A"
            Label {
                anchors.fill: parent
                anchors.margins: 8
                text: albumBackend.serviceMessage
                elide: Text.ElideMiddle
                color: albumBackend.serviceReady ? "#B8E8D4" : "#FFD0D0"
                verticalAlignment: Text.AlignVCenter
                font.pixelSize: 12
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10

            Rectangle {
                Layout.preferredWidth: 230
                Layout.fillHeight: true
                radius: 14
                color: "#151F2D"
                border.color: "#2C4058"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    Label { text: "Library"; font.pixelSize: 17; font.weight: 700; color: "#EAF1FF" }
                    TextField { Layout.fillWidth: true; placeholderText: "Search folders" }
                    Button { Layout.fillWidth: true; text: "All Photos"; onClicked: settingsPage = false }
                    Button { Layout.fillWidth: true; text: "Recent Imports"; onClicked: settingsPage = false }
                    Button { Layout.fillWidth: true; text: "Collections"; onClicked: settingsPage = false }
                    Button { Layout.fillWidth: true; text: "Settings"; onClicked: settingsPage = true }
                    Item { Layout.fillHeight: true }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 10

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    radius: 12
                    color: "#162435"
                    border.color: "#2F425A"
                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 9
                        Label { text: "Browser"; color: "#F2F7FF"; font.pixelSize: 17; font.weight: 700 }
                        Label { text: "Responsive thumbnail grid"; color: "#99AFD1"; font.pixelSize: 12 }
                        Item { Layout.fillWidth: true }
                        Button { text: "Grid"; checkable: true; checked: gridMode; onClicked: gridMode = true }
                        Button { text: "List"; checkable: true; checked: !gridMode; onClicked: gridMode = false }
                    }
                }

                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: settingsPage ? 1 : 0

                    Rectangle {
                        radius: 14
                        color: "#151F2D"
                        border.color: "#2E425A"

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: "Thumbnails"; color: "#EDF4FF"; font.pixelSize: 16; font.weight: 700 }
                                Item { Layout.fillWidth: true }
                                Label { text: albumBackend.filterInfo; color: "#99AFD1"; font.pixelSize: 12 }
                            }

                            Loader {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                active: albumBackend.shownCount > 0
                                sourceComponent: gridMode ? gridComp : listComp
                            }

                            Column {
                                Layout.fillWidth: true
                                visible: albumBackend.shownCount === 0
                                spacing: 8
                                Label { text: "No Photos Yet"; color: "#EDF4FF"; font.pixelSize: 22; font.weight: 700 }
                                Label { text: "Import your first folder to start thumbnail generation and RAW adjustments."; color: "#9BB1D1"; font.pixelSize: 12 }
                                Button { text: "Import Photos"; onClicked: importDialog.open() }
                            }
                        }
                    }

                    Rectangle {
                        radius: 14
                        color: "#162131"
                        border.color: "#2E425A"
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            Label { text: "Settings"; color: "#EDF4FF"; font.pixelSize: 20; font.weight: 700 }
                            Label { text: "Window #0E131A  Text #E8EEFA  Accent #5AA2FF"; color: "#99AFD1"; font.pixelSize: 12 }
                            Label { text: "Qt Quick renderer is hardware accelerated."; color: "#99AFD1"; font.pixelSize: 12 }
                            Item { Layout.fillHeight: true }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: inspectorVisible && !settingsPage ? 350 : 0
                Behavior on Layout.preferredWidth { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }
                radius: 14
                color: "#121B28"
                border.color: "#2D4058"
                clip: true
                visible: Layout.preferredWidth > 10

                ScrollView {
                    anchors.fill: parent
                    anchors.margins: 10
                    ColumnLayout {
                        width: parent.width
                        spacing: 10

                        Rectangle {
                            Layout.fillWidth: true
                            radius: 12
                            color: "#182332"
                            border.color: "#2F425A"
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 10
                                RowLayout {
                                    Layout.fillWidth: true
                                    Label { text: "Filters"; color: "#EEF5FF"; font.pixelSize: 17; font.weight: 700 }
                                    Item { Layout.fillWidth: true }
                                    Button { text: drawerOpen ? "Collapse" : "Expand"; onClicked: drawerOpen = !drawerOpen }
                                }

                                ColumnLayout {
                                    visible: drawerOpen
                                    Layout.fillWidth: true
                                    TextField { id: quickSearch; Layout.fillWidth: true; placeholderText: "Search filename, camera model, tags"; onAccepted: applyFilter() }

                                    RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: "Rules"; color: "#EDF4FF" }
                                        Item { Layout.fillWidth: true }
                                        ComboBox {
                                            id: joinCombo
                                            model: [ { text: "Match all rules", value: 0 }, { text: "Match any rule", value: 1 } ]
                                            textRole: "text"
                                            valueRole: "value"
                                        }
                                    }

                                    ListView {
                                        id: rules
                                        Layout.fillWidth: true
                                        Layout.preferredHeight: 280
                                        clip: true
                                        spacing: 6
                                        model: albumBackend.filterRules
                                        delegate: Rectangle {
                                            required property int index
                                            required property int fieldValue
                                            required property int opValue
                                            required property string valueText
                                            required property string value2Text
                                            required property bool showSecondValue
                                            required property string placeholder
                                            required property var opOptions
                                            width: rules.width
                                            height: showSecondValue ? 96 : 66
                                            radius: 10
                                            color: "#101A27"
                                            border.color: "#2D4159"
                                            Column {
                                                anchors.fill: parent
                                                anchors.margins: 6
                                                spacing: 4
                                                Row {
                                                    width: parent.width
                                                    spacing: 4
                                                    ComboBox { width: parent.width * 0.43; model: albumBackend.fieldOptions; textRole: "text"; valueRole: "value"; currentIndex: root.idxByValue(model, fieldValue); onActivated: albumBackend.setRuleField(index, currentValue) }
                                                    ComboBox { width: parent.width * 0.35; model: opOptions; textRole: "text"; valueRole: "value"; currentIndex: root.idxByValue(opOptions, opValue); onActivated: albumBackend.setRuleOp(index, currentValue) }
                                                    Button { width: parent.width * 0.18; text: "X"; onClicked: albumBackend.removeRule(index) }
                                                }
                                                Row {
                                                    width: parent.width
                                                    spacing: 4
                                                    TextField { width: showSecondValue ? parent.width * 0.49 : parent.width; text: valueText; placeholderText: placeholder; onTextEdited: albumBackend.setRuleValue(index, text) }
                                                    TextField { visible: showSecondValue; width: parent.width * 0.49; text: value2Text; placeholderText: placeholder; onTextEdited: albumBackend.setRuleValue2(index, text) }
                                                }
                                            }
                                        }
                                    }

                                    Button { text: "Add rule"; onClicked: albumBackend.addRule() }
                                    RowLayout {
                                        Layout.fillWidth: true
                                        Button { text: "Clear"; onClicked: { quickSearch.text = ""; joinCombo.currentIndex = 0; albumBackend.clearFilters() } }
                                        Item { Layout.fillWidth: true }
                                        Button { text: "Apply"; onClicked: applyFilter() }
                                    }
                                    Label { text: albumBackend.filterInfo; color: "#9EB4D4"; font.pixelSize: 12 }
                                    Label { visible: albumBackend.validationError.length > 0; text: albumBackend.validationError; color: "#FFB8B8"; wrapMode: Text.WordWrap; font.pixelSize: 12 }
                                    Label { visible: albumBackend.sqlPreview.length > 0; text: albumBackend.sqlPreview; wrapMode: Text.WrapAnywhere; color: "#DBE9FF"; font.family: "Consolas"; font.pixelSize: 11 }
                                }
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            radius: 12
                            color: "#172332"
                            border.color: "#2E425A"
                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 10
                                spacing: 8

                                RowLayout {
                                    Layout.fillWidth: true
                                    Label {
                                        text: "Editor"
                                        color: "#EEF5FF"
                                        font.pixelSize: 17
                                        font.weight: 700
                                    }
                                    Item { Layout.fillWidth: true }
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: "The full OpenGL editor opens in a separate dialog window."
                                    color: "#9EB4D4"
                                    font.pixelSize: 12
                                    wrapMode: Text.WordWrap
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: albumBackend.editorStatus
                                    color: "#D8E7FF"
                                    font.pixelSize: 12
                                    wrapMode: Text.WordWrap
                                }

                                Label {
                                    visible: albumBackend.editorTitle.length > 0
                                    Layout.fillWidth: true
                                    text: albumBackend.editorTitle
                                    color: "#D8E7FF"
                                    font.pixelSize: 12
                                    elide: Text.ElideRight
                                }

                                Label {
                                    Layout.fillWidth: true
                                    text: albumBackend.editorActive
                                          ? "Editor window is open. Close that window to save edits."
                                          : "Click Edit on a thumbnail to open the full editor dialog."
                                    color: "#8EA4C6"
                                    font.pixelSize: 12
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 58
            radius: 12
            color: "#172332"
            border.color: "#2F425A"
            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                Label { Layout.fillWidth: true; text: albumBackend.taskStatus; color: "#99AFD1" }
                ProgressBar { Layout.preferredWidth: 240; value: albumBackend.taskProgress / 100.0 }
                Button { visible: albumBackend.taskCancelVisible; text: "Cancel"; onClicked: albumBackend.cancelImport() }
            }
        }
    }

    Component {
        id: gridComp
        GridView {
            anchors.fill: parent
            model: albumBackend.thumbnails
            clip: true
            cellWidth: 242
            cellHeight: 186
            delegate: Rectangle {
                required property int elementId
                required property int imageId
                required property string fileName
                required property string cameraModel
                required property int iso
                required property string aperture
                required property string captureDate
                required property int rating
                required property string accent
                required property string thumbUrl
                width: 230
                height: 172
                radius: 12
                color: "#111A27"
                border.color: albumBackend.editorActive && albumBackend.editorElementId === elementId ? "#5AA2FF" : "#2E425A"
                Item {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.margins: 8
                    height: 92
                    clip: true
                    Rectangle {
                        anchors.fill: parent
                        radius: 9
                        visible: thumbUrl.length === 0
                        gradient: Gradient {
                            GradientStop { position: 0.0; color: Qt.lighter(accent, 1.3) }
                            GradientStop { position: 1.0; color: Qt.darker(accent, 1.6) }
                        }
                    }
                    Image {
                        anchors.fill: parent
                        source: thumbUrl
                        visible: thumbUrl.length > 0
                        asynchronous: true
                        cache: false
                        fillMode: Image.PreserveAspectFit
                    }
                    Button {
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 6
                        text: "Edit"
                        onClicked: albumBackend.openEditor(elementId, imageId)
                    }
                }
                Column { anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom; anchors.margins: 8; spacing: 2
                    Label { text: fileName; color: "#EDF4FF"; font.pixelSize: 12; elide: Text.ElideRight; width: parent.width }
                    Label { text: cameraModel + " | ISO " + iso + " | f/" + aperture; color: "#9DB3D4"; font.pixelSize: 11; elide: Text.ElideRight; width: parent.width }
                    Label { text: captureDate + " | rating " + rating + "/5"; color: "#8299BB"; font.pixelSize: 10; elide: Text.ElideRight; width: parent.width }
                }
            }
        }
    }

    Component {
        id: listComp
        ListView {
            anchors.fill: parent
            model: albumBackend.thumbnails
            clip: true
            spacing: 8
            delegate: Rectangle {
                required property int elementId
                required property int imageId
                required property string fileName
                required property string cameraModel
                required property string extension
                required property int iso
                required property string aperture
                required property string focalLength
                required property string captureDate
                required property int rating
                required property string tags
                required property string accent
                required property string thumbUrl
                width: ListView.view.width
                height: 84
                radius: 12
                color: "#111A27"
                border.color: albumBackend.editorActive && albumBackend.editorElementId === elementId ? "#5AA2FF" : "#2E425A"
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 8
                    Item {
                        Layout.preferredWidth: 96
                        Layout.fillHeight: true
                        clip: true
                        Rectangle {
                            anchors.fill: parent
                            radius: 8
                            visible: thumbUrl.length === 0
                            gradient: Gradient {
                                GradientStop { position: 0.0; color: Qt.lighter(accent, 1.3) }
                                GradientStop { position: 1.0; color: Qt.darker(accent, 1.7) }
                            }
                        }
                        Image {
                            anchors.fill: parent
                            source: thumbUrl
                            visible: thumbUrl.length > 0
                            asynchronous: true
                            cache: false
                            fillMode: Image.PreserveAspectFit
                        }
                    }
                    ColumnLayout { Layout.fillWidth: true
                        Label { Layout.fillWidth: true; text: fileName; color: "#EDF4FF"; font.pixelSize: 13; elide: Text.ElideRight }
                        Label { Layout.fillWidth: true; text: cameraModel + " | " + extension + " | ISO " + iso + " | f/" + aperture + " | " + focalLength + "mm"; color: "#9DB3D4"; font.pixelSize: 11; elide: Text.ElideRight }
                        Label { Layout.fillWidth: true; text: captureDate + " | tags: " + tags; color: "#8299BB"; font.pixelSize: 10; elide: Text.ElideRight }
                    }
                    ColumnLayout {
                        spacing: 4
                        Label { text: rating + "/5"; color: "#D6E5FF"; font.pixelSize: 12; font.weight: 700; horizontalAlignment: Text.AlignHCenter }
                        Button { text: "Edit"; onClicked: albumBackend.openEditor(elementId, imageId) }
                    }
                }
            }
        }
    }
}
