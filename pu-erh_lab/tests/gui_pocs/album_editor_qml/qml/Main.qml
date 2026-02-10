import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Effects

ApplicationWindow {
    id: root
    width: 1460
    height: 900
    visible: true
    visibility: Window.Maximized
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
    property string pendingNewProjectFolderUrl: ""
    property string defaultNewProjectName: "album_editor_project"
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
        id: loadProjectDialog
        title: "Select Project Package or Metadata JSON"
        fileMode: FileDialog.OpenFile
        nameFilters: [
            "Packed Project (*.puerhproj)",
            "Project Metadata (*.json)",
            "All Files (*)"
        ]
        onAccepted: {
            albumBackend.loadProject(selectedFile.toString())
        }
    }

    FolderDialog {
        id: createProjectFolderDialog
        title: "Select Parent Folder for New Project"
        onAccepted: {
            root.pendingNewProjectFolderUrl = selectedFolder.toString()
            createProjectNameField.text = root.defaultNewProjectName
            createProjectNameDialog.open()
        }
    }

    Dialog {
        id: createProjectNameDialog
        modal: true
        title: "Name New Project"
        standardButtons: Dialog.NoButton
        closePolicy: Popup.CloseOnEscape
        x: Math.round((root.width - width) / 2)
        y: Math.round((root.height - height) / 2)

        function submitCreateProject() {
            const trimmed = createProjectNameField.text.trim()
            if (trimmed.length === 0 || root.pendingNewProjectFolderUrl.length === 0) {
                return
            }
            root.defaultNewProjectName = trimmed
            albumBackend.createProjectInFolderNamed(root.pendingNewProjectFolderUrl, trimmed)
            root.pendingNewProjectFolderUrl = ""
            createProjectNameDialog.close()
        }

        onOpened: {
            createProjectNameField.forceActiveFocus()
            createProjectNameField.selectAll()
        }

        onClosed: {
            root.pendingNewProjectFolderUrl = ""
        }

        contentItem: ColumnLayout {
            width: 420
            spacing: 12

            Label {
                Layout.fillWidth: true
                text: "Choose the project package name. The app will create a single .puerhproj file."
                wrapMode: Text.WordWrap
                color: "#E3DFDB"
            }

            TextField {
                id: createProjectNameField
                Layout.fillWidth: true
                placeholderText: "Project name"
                onAccepted: createProjectNameDialog.submitCreateProject()
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                Item { Layout.fillWidth: true }

                Button {
                    text: "Cancel"
                    onClicked: createProjectNameDialog.close()
                }

                Button {
                    text: "Create"
                    enabled: createProjectNameField.text.trim().length > 0
                    onClicked: createProjectNameDialog.submitCreateProject()
                }
            }
        }
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
        function onProjectChanged() {
            root.clearSelectedImages()
            root.clearExportQueue()
            settingsPage = false
        }
        function onFolderSelectionChanged() {
            root.clearSelectedImages()
        }
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

    Item {
        id: mainContent
        anchors.fill: parent

        Rectangle {
            anchors.fill: parent
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
                Button { text: "Load"; enabled: !albumBackend.projectLoading; onClicked: loadProjectDialog.open() }
                Button { text: "New"; enabled: !albumBackend.projectLoading; onClicked: createProjectFolderDialog.open() }
                Button { text: "Save"; enabled: albumBackend.serviceReady && !albumBackend.projectLoading; onClicked: albumBackend.saveProject() }
                Button { text: "Import"; enabled: albumBackend.serviceReady && !albumBackend.projectLoading; onClicked: importDialog.open() }
                Button { text: "Add Selected (" + root.selectedCount + ")"; enabled: albumBackend.serviceReady && !albumBackend.projectLoading && root.selectedCount > 0; onClicked: root.addSelectedToExportQueue() }
                Button {
                    text: "Export Queue (" + root.exportQueueCount + ")"
                    enabled: albumBackend.serviceReady && !albumBackend.projectLoading && (albumBackend.shownCount > 0 || root.exportQueueCount > 0)
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
            color: albumBackend.projectLoading ? "#1E1D22" : (albumBackend.serviceReady ? "#38373C" : "#1E1D22")
            border.color: albumBackend.projectLoading ? "#92BCE1" : (albumBackend.serviceReady ? "#A1AC9B" : "#7B7D7C")
            Label {
                anchors.fill: parent
                anchors.margins: 8
                text: albumBackend.serviceMessage
                elide: Text.ElideMiddle
                color: albumBackend.projectLoading ? "#92BCE1" : (albumBackend.serviceReady ? "#E3DFDB" : "#7B7D7C")
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
                    Label { text: albumBackend.currentFolderPath; color: "#7B7D7C"; font.pixelSize: 11; elide: Text.ElideMiddle; Layout.fillWidth: true }

                    TextField {
                        id: folderSearchField
                        Layout.fillWidth: true
                        placeholderText: "Search folders"
                    }

                    TextField {
                        id: createFolderField
                        Layout.fillWidth: true
                        placeholderText: "New folder name"
                        enabled: albumBackend.serviceReady && !albumBackend.projectLoading
                        onAccepted: {
                            if (text.trim().length === 0) {
                                return
                            }
                            albumBackend.createFolder(text)
                            text = ""
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Button {
                            Layout.fillWidth: true
                            text: "Add Folder"
                            enabled: albumBackend.serviceReady
                                     && !albumBackend.projectLoading
                                     && createFolderField.text.trim().length > 0
                            onClicked: {
                                albumBackend.createFolder(createFolderField.text)
                                createFolderField.text = ""
                            }
                        }
                        Button {
                            Layout.fillWidth: true
                            text: "Delete"
                            enabled: albumBackend.serviceReady
                                     && !albumBackend.projectLoading
                                     && albumBackend.currentFolderId !== 0
                            onClicked: albumBackend.deleteFolder(albumBackend.currentFolderId)
                        }
                    }

                    ListView {
                        id: folderList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 4
                        model: albumBackend.folders
                        delegate: Item {
                            required property int folderId
                            required property string name
                            required property int depth
                            required property string path
                            width: ListView.view.width
                            height: folderButton.visible ? 32 : 0
                            visible: folderSearchField.text.trim().length === 0
                                     || name.toLowerCase().indexOf(folderSearchField.text.trim().toLowerCase()) >= 0

                            Button {
                                id: folderButton
                                anchors.fill: parent
                                text: name
                                checkable: true
                                checked: folderId === albumBackend.currentFolderId
                                leftPadding: 10 + depth * 14
                                onClicked: {
                                    settingsPage = false
                                    albumBackend.selectFolder(folderId)
                                }
                            }
                        }
                    }
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
                                Label {
                                    text: albumBackend.serviceReady ? "No Photos Yet" : "Open or Create a Project"
                                    color: "#E3DFDB"
                                    font.pixelSize: 22
                                    font.weight: 700
                                }
                                Label {
                                    text: albumBackend.serviceReady
                                          ? "Import your first folder to start thumbnail generation and RAW adjustments."
                                          : "Use Load or New in the header to choose the database/metadata JSON files."
                                    color: "#7B7D7C"
                                    font.pixelSize: 12
                                }
                                Button {
                                    text: albumBackend.serviceReady ? "Import Photos" : "Load Project"
                                    onClicked: {
                                        if (albumBackend.serviceReady) {
                                            importDialog.open()
                                        } else {
                                            loadProjectDialog.open()
                                        }
                                    }
                                }
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

    }

    Item {
        anchors.fill: parent
        visible: !albumBackend.serviceReady && !albumBackend.projectLoading
        z: 30

        MultiEffect {
            anchors.fill: parent
            source: mainContent
            blurEnabled: true
            blur: 0.6
            blurMax: 64
            saturation: -0.2
        }

        Rectangle {
            anchors.fill: parent
            color: "#C01E1D22"
        }

        MouseArea { anchors.fill: parent; hoverEnabled: true }

        Rectangle {
            anchors.centerIn: parent
            width: Math.min(parent.width - 36, 700)
            height: dialogContent.implicitHeight + 36
            radius: 14
            color: "#1E1D22"
            border.color: "#4D4C51"

            ColumnLayout {
                id: dialogContent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 18
                spacing: 12

                Label {
                    text: "Open Project"
                    font.pixelSize: 24
                    font.weight: 700
                    color: "#E3DFDB"
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: "Every boot asks for a project. Load a packed .puerhproj file or a metadata JSON/database pair, or create a new project package."
                    color: "#7B7D7C"
                    font.pixelSize: 13
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: albumBackend.serviceMessage
                    color: "#E3DFDB"
                    font.pixelSize: 12
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10

                    Button {
                        Layout.fillWidth: true
                        text: "Load Existing Project"
                        onClicked: loadProjectDialog.open()
                    }
                    Button {
                        Layout.fillWidth: true
                        text: "Create New Project"
                        onClicked: createProjectFolderDialog.open()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    Item { Layout.fillWidth: true }
                    Button {
                        text: "Exit"
                        onClicked: Qt.quit()
                    }
                }
            }
        }
    }

    Item {
        anchors.fill: parent
        visible: albumBackend.projectLoading
        z: 40

        MultiEffect {
            anchors.fill: parent
            source: mainContent
            blurEnabled: true
            blur: 0.6
            blurMax: 64
            saturation: -0.2
        }

        Rectangle {
            anchors.fill: parent
            color: "#C01E1D22"
        }

        MouseArea { anchors.fill: parent; hoverEnabled: true }

        Rectangle {
            anchors.centerIn: parent
            width: Math.min(parent.width - 36, 520)
            height: loadingContent.implicitHeight + 30
            radius: 14
            color: "#1E1D22"
            border.color: "#4D4C51"

            ColumnLayout {
                id: loadingContent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 16
                spacing: 10

                Label {
                    text: "Loading Project"
                    font.pixelSize: 21
                    font.weight: 700
                    color: "#E3DFDB"
                }
                BusyIndicator {
                    running: albumBackend.projectLoading
                    Layout.alignment: Qt.AlignHCenter
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: albumBackend.projectLoadingMessage.length > 0
                          ? albumBackend.projectLoadingMessage
                          : "Preparing database and metadata..."
                    color: "#7B7D7C"
                    font.pixelSize: 12
                    horizontalAlignment: Text.AlignHCenter
                }
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
