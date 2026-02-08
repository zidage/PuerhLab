import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: root
    width: 1460
    height: 900
    visible: true
    title: "pu-erh_lab - Album + Editor (QML POC)"

    // Dark palette:
    // Bases: #1E1D22 #38373C #4D4C51  Text: #E3DFDB #7B7D7C  Accents: #A1AC9B #92BCE1 #E03D46
    Material.theme: Material.Dark
    Material.accent: "#92BCE1"
    color: "#1E1D22"

    property bool settingsPage: false
    property bool inspectorVisible: true
    property bool gridMode: true
    property var selectedImagesById: ({})
    property var exportQueueById: ({})
    property var exportPreviewRows: []
    readonly property int selectedCount: Object.keys(selectedImagesById).length
    readonly property int exportQueueCount: Object.keys(exportQueueById).length

    function keyForElement(elementId) {
        return String(Number(elementId))
    }

    function isImageSelected(elementId) {
        return Object.prototype.hasOwnProperty.call(
            selectedImagesById, keyForElement(elementId))
    }

    function isImageQueued(elementId) {
        return Object.prototype.hasOwnProperty.call(
            exportQueueById, keyForElement(elementId))
    }

    function setImageSelected(elementId, imageId, fileName, selected) {
        const key = keyForElement(elementId)
        const already = Object.prototype.hasOwnProperty.call(selectedImagesById, key)
        if (selected === already) {
            return
        }

        const next = Object.assign({}, selectedImagesById)
        if (selected) {
            next[key] = {
                elementId: Number(elementId),
                imageId: Number(imageId),
                fileName: fileName ? fileName : "(unnamed)"
            }
        } else {
            delete next[key]
        }
        selectedImagesById = next
    }

    function clearSelectedImages() {
        selectedImagesById = ({})
    }

    function addSelectedToExportQueue() {
        const selected = Object.values(selectedImagesById)
        if (selected.length === 0) {
            return
        }

        const next = Object.assign({}, exportQueueById)
        for (let i = 0; i < selected.length; ++i) {
            const item = selected[i]
            next[keyForElement(item.elementId)] = item
        }
        exportQueueById = next
        clearSelectedImages()
        refreshExportPreview()
    }

    function clearExportQueue() {
        exportQueueById = ({})
        refreshExportPreview()
    }

    function exportQueueTargets() {
        const rows = Object.values(exportQueueById)
        const targets = []
        for (let i = 0; i < rows.length; ++i) {
            targets.push({
                elementId: rows[i].elementId,
                imageId: rows[i].imageId
            })
        }
        return targets
    }

    function refreshExportPreview() {
        const src = Object.values(exportQueueById)
        src.sort((a, b) => String(a.fileName).localeCompare(String(b.fileName)))
        const next = []
        const previewCount = Math.min(12, src.length)
        for (let i = 0; i < previewCount; ++i) {
            const item = src[i]
            next.push({
                label: "Image #" + item.imageId + "  Sleeve #" + item.elementId + "  " + item.fileName
            })
        }
        if (src.length > previewCount) {
            next.push({ label: "... and " + (src.length - previewCount) + " more" })
        }
        exportPreviewRows = next
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

    AlbumExportDialog {
        id: exportDialog
        selectedCount: root.selectedCount
        exportQueueCount: root.exportQueueCount
        exportPreviewRows: root.exportPreviewRows
        onAddSelectedToQueueRequested: root.addSelectedToExportQueue()
        onClearQueueRequested: root.clearExportQueue()
        onEnsurePreviewRequested: root.refreshExportPreview()
        onStartExportRequested: function(outDir, format, resizeEnabled, maxSide, quality, bitDepth, pngLevel, tiffComp) {
            albumBackend.startExportWithOptionsForTargets(
                outDir,
                format,
                resizeEnabled,
                maxSide,
                quality,
                bitDepth,
                pngLevel,
                tiffComp,
                root.exportQueueTargets())
        }
    }

    Connections {
        target: albumBackend
        function onThumbnailsChanged() {
            if (exportDialog.visible) {
                refreshExportPreview()
            }
        }
        function onExportStateChanged() {
            if (!exportDialog.visible) {
                return
            }
            if (exportDialog.exportTriggered
                    && !albumBackend.exportInFlight
                    && albumBackend.exportCompleted > 0) {
                exportDialog.close()
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        z: -1
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#1E1D22" }
            GradientStop { position: 1.0; color: "#38373C" }
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
            color: "#38373C"
            border.color: "#4D4C51"

            RowLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 10
                anchors.rightMargin: 10
                Label { text: "PuerhLab"; font.pixelSize: 19; font.weight: 700; color: "#E3DFDB" }
                Button { text: "Library"; checkable: true; checked: !settingsPage; onClicked: settingsPage = false }
                Button { text: "Settings"; checkable: true; checked: settingsPage; onClicked: settingsPage = true }
                TextField { Layout.fillWidth: true; placeholderText: "Search photos" }
                Button { text: "Import"; onClicked: importDialog.open() }
                Button { text: "Add Selected (" + root.selectedCount + ")"; enabled: root.selectedCount > 0; onClicked: root.addSelectedToExportQueue() }
                Button {
                    text: "Export Queue (" + root.exportQueueCount + ")"
                    enabled: albumBackend.shownCount > 0 || root.exportQueueCount > 0
                    onClicked: {
                        refreshExportPreview()
                        exportDialog.open()
                    }
                }
                Button { text: "Inspector"; checkable: true; checked: inspectorVisible; onToggled: inspectorVisible = checked }
            }
        }

        Rectangle {
            visible: albumBackend.serviceMessage.length > 0
            Layout.fillWidth: true
            Layout.preferredHeight: 34
            radius: 10
            color: albumBackend.serviceReady ? "#38373C" : "#1E1D22"
            border.color: albumBackend.serviceReady ? "#A1AC9B" : "#7B7D7C"
            Label {
                anchors.fill: parent
                anchors.margins: 8
                text: albumBackend.serviceMessage
                elide: Text.ElideMiddle
                color: albumBackend.serviceReady ? "#E3DFDB" : "#7B7D7C"
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
                color: "#1E1D22"
                border.color: "#4D4C51"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    Label { text: "Library"; font.pixelSize: 17; font.weight: 700; color: "#E3DFDB" }
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
                    color: "#38373C"
                    border.color: "#4D4C51"
                    RowLayout {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.leftMargin: 9
                        anchors.rightMargin: 9
                        Label { text: "Browser"; color: "#E3DFDB"; font.pixelSize: 17; font.weight: 700 }
                        Label { text: "Responsive thumbnail grid"; color: "#7B7D7C"; font.pixelSize: 12 }
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
                        color: "#1E1D22"
                        border.color: "#4D4C51"

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: "Album"; color: "#E3DFDB"; font.pixelSize: 16; font.weight: 700 }
                                Item { Layout.fillWidth: true }
                                Label { text: albumBackend.filterInfo; color: "#7B7D7C"; font.pixelSize: 12 }
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
                                Label { text: "No Photos Yet"; color: "#E3DFDB"; font.pixelSize: 22; font.weight: 700 }
                                Label { text: "Import your first folder to start thumbnail generation and RAW adjustments."; color: "#7B7D7C"; font.pixelSize: 12 }
                                Button { text: "Import Photos"; onClicked: importDialog.open() }
                            }
                        }
                    }

                    Rectangle {
                        radius: 14
                        color: "#1E1D22"
                        border.color: "#4D4C51"
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            Label { text: "Settings"; color: "#E3DFDB"; font.pixelSize: 20; font.weight: 700 }
                            Label { text: "Window #1E1D22  Text #E3DFDB  Accent #92BCE1"; color: "#7B7D7C"; font.pixelSize: 12 }
                            Label { text: "Qt Quick renderer is hardware accelerated."; color: "#7B7D7C"; font.pixelSize: 12 }
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
                color: "#1E1D22"
                border.color: "#4D4C51"
                clip: true
                visible: Layout.preferredWidth > 10

                InspectorPanel {
                    anchors.fill: parent
                    anchors.margins: 10
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 58
            radius: 12
            color: "#38373C"
            border.color: "#4D4C51"
            RowLayout {
                anchors.fill: parent
                anchors.margins: 10
                Label { Layout.fillWidth: true; text: albumBackend.taskStatus; color: "#7B7D7C" }
                ProgressBar { Layout.preferredWidth: 240; value: albumBackend.taskProgress / 100.0 }
                Button { visible: albumBackend.taskCancelVisible; text: "Cancel"; onClicked: albumBackend.cancelImport() }
            }
        }
    }

    Component {
        id: gridComp
        ThumbnailGridView {
            selectedImagesById: root.selectedImagesById
            exportQueueById: root.exportQueueById
            onImageSelectionChanged: function(elementId, imageId, fileName, selected) {
                root.setImageSelected(elementId, imageId, fileName, selected)
            }
        }
    }

    Component {
        id: listComp
        ThumbnailListView {
            selectedImagesById: root.selectedImagesById
            exportQueueById: root.exportQueueById
            onImageSelectionChanged: function(elementId, imageId, fileName, selected) {
                root.setImageSelected(elementId, imageId, fileName, selected)
            }
        }
    }
}
