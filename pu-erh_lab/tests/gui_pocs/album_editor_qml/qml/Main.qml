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

    // Theme palette — borderless, luminance-separated zones
    readonly property color toneGold: "#FCC704"
    readonly property color toneWine: "#8A0526"
    readonly property color toneSteel: "#4A4A4A"
    readonly property color toneGraphite: "#1A1A1A"
    readonly property color toneMist: "#E6E6E6"
    readonly property color toneAmber: "#FCC704"
    readonly property color toneRose: "#FCC704"

    readonly property color colBgDeep: "#141414"        // center content — darkest "stage"
    readonly property color colBgBase: "#1F1F1F"        // sunken inputs
    readonly property color colBgPanel: "#2B2B2B"       // side panels & header/footer
    readonly property color colBgCanvas: "#111111"      // gap / outer canvas behind blocks
    readonly property int panelRadius: 8                // uniform rounded-corner radius
    readonly property color colBorder: "transparent"     // NO borders by default
    readonly property color colText: toneMist
    readonly property color colTextMuted: "#888888"
    readonly property color colAccentPrimary: toneGold
    readonly property color colAccentSecondary: toneGold
    readonly property color colAccentSoft: toneGold
    readonly property color colDanger: toneWine
    readonly property color colDangerTint: Qt.rgba(138 / 255, 5 / 255, 38 / 255, 0.32)
    readonly property color colSelectedTint: Qt.rgba(252 / 255, 199 / 255, 4 / 255, 0.18)
    readonly property color colHover: "#333333"          // subtle hover tint
    readonly property color colOverlay: "#C0121212"

    Material.theme: Material.Dark
    Material.primary: root.colAccentSecondary
    Material.accent: root.colAccentPrimary
    Material.background: root.colBgPanel
    Material.foreground: root.colText
    color: root.colBgPanel

    property bool settingsPage: false
    property bool inspectorVisible: true
    property bool gridMode: true
    property bool selectionMode: false
    property string pendingNewProjectFolderUrl: ""
    property string defaultNewProjectName: "album_editor_project"
    readonly property bool backendInteractive: albumBackend.serviceReady && !albumBackend.projectLoading
    readonly property var selectedImagesById: selectionState.selectedImagesById
    readonly property var exportQueueById: selectionState.exportQueueById
    readonly property var exportPreviewRows: selectionState.exportPreviewRows
    readonly property int selectedCount: selectionState.selectedCount
    readonly property int exportQueueCount: selectionState.exportQueueCount

    QtObject {
        id: selectionState
        property var selectedImagesById: ({})
        property var exportQueueById: ({})
        property var exportPreviewRows: []
        readonly property int selectedCount: Object.keys(selectedImagesById).length
        readonly property int exportQueueCount: Object.keys(exportQueueById).length

        function keyForElement(elementId) {
            return String(Number(elementId))
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

        function replaceSelectedImages(items) {
            const next = {}
            for (let i = 0; i < items.length; ++i) {
                const item = items[i]
                const key = keyForElement(item.elementId)
                next[key] = {
                    elementId: Number(item.elementId),
                    imageId: Number(item.imageId),
                    fileName: item.fileName ? item.fileName : "(unnamed)"
                }
            }
            selectedImagesById = next
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
                color: root.colText
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
        onAddSelectedToQueueRequested: selectionState.addSelectedToExportQueue()
        onClearQueueRequested: selectionState.clearExportQueue()
        onEnsurePreviewRequested: selectionState.refreshExportPreview()
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
                selectionState.exportQueueTargets())
        }
    }

    Connections {
        target: albumBackend
        function onProjectChanged() {
            selectionState.clearSelectedImages()
            selectionState.clearExportQueue()
            settingsPage = false
        }
        function onFolderSelectionChanged() {
            selectionState.clearSelectedImages()
        }
        function onThumbnailsChanged() {
            if (exportDialog.visible) {
                selectionState.refreshExportPreview()
            }
        }
        function onExportStateChanged() {
        }
    }

    Item {
        id: mainContent
        anchors.fill: parent

        Rectangle {
            anchors.fill: parent
            color: root.colBgCanvas
        }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 3
        spacing: 3

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 50
            radius: root.panelRadius
            color: root.colBgPanel
            border.width: 0

            RowLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 16
                anchors.rightMargin: 16
                spacing: 4
                Label { text: "PuerhLab"; font.pixelSize: 19; font.weight: 700; color: root.colAccentPrimary }
                Item { Layout.preferredWidth: 8 }
                Button { text: "Load"; enabled: !albumBackend.projectLoading; onClicked: loadProjectDialog.open() }
                Button { text: "New"; enabled: !albumBackend.projectLoading; onClicked: createProjectFolderDialog.open() }
                Button { text: "Save"; enabled: root.backendInteractive; onClicked: albumBackend.saveProject() }
                Item { Layout.preferredWidth: 4 }
                Button {
                    text: "Import"
                    enabled: root.backendInteractive
                    Material.background: root.colAccentSoft
                    Material.foreground: root.colBgDeep
                    onClicked: importDialog.open()
                }
                Item { Layout.fillWidth: true }
                Button {
                    text: "Add Selected (" + root.selectedCount + ")"
                    enabled: root.backendInteractive && root.selectedCount > 0
                    Material.background: root.colAccentPrimary
                    Material.foreground: root.colBgDeep
                    onClicked: selectionState.addSelectedToExportQueue()
                }
                Button {
                    text: "Export Queue (" + root.exportQueueCount + ")"
                    enabled: root.backendInteractive && (albumBackend.shownCount > 0 || root.exportQueueCount > 0)
                    Material.background: root.colAccentSecondary
                    Material.foreground: root.colBgDeep
                    onClicked: {
                        selectionState.refreshExportPreview()
                        exportDialog.open()
                    }
                }
                Item { Layout.preferredWidth: 8 }
                Button { text: "Library"; checkable: true; checked: !settingsPage; onClicked: settingsPage = false }
                Button { text: "Settings"; checkable: true; checked: settingsPage; onClicked: settingsPage = true }
                Button { text: "Inspector"; checkable: true; checked: inspectorVisible; onToggled: inspectorVisible = checked }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 3

            Rectangle {
                Layout.preferredWidth: 250
                Layout.fillHeight: true
                radius: root.panelRadius
                color: root.colBgPanel
                border.width: 0
                clip: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 10

                    // ── Header ──
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Label {
                            text: "\u{1F4C1}"
                            font.pixelSize: 18
                        }
                        Label {
                            text: "Library"
                            font.pixelSize: 17
                            font.weight: 700
                            color: root.colText
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: folderList.count + " folders"
                            color: root.colAccentSoft
                            font.pixelSize: 11
                        }
                    }

                    Label {
                        text: albumBackend.currentFolderPath
                        color: root.colTextMuted
                        font.pixelSize: 11
                        elide: Text.ElideMiddle
                        Layout.fillWidth: true
                    }

                    // ── Search ──
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 36
                        radius: 6
                        color: root.colBgBase
                        border.width: 0
                        Behavior on color { ColorAnimation { duration: 150 } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 8
                            anchors.rightMargin: 8
                            spacing: 6
                            Label { text: "\u{1F50D}"; font.pixelSize: 13; color: root.colTextMuted }
                            TextField {
                                id: folderSearchField
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                placeholderText: "Search folders..."
                                background: Item {}
                                color: root.colText
                                font.pixelSize: 12
                            }
                        }
                    }

                    // ── New-folder row ──
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 36
                        radius: 6
                        color: root.colBgBase
                        border.width: 0
                        Behavior on color { ColorAnimation { duration: 150 } }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 8
                            anchors.rightMargin: 4
                            spacing: 4
                            Label { text: "+"; font.pixelSize: 16; font.weight: 700; color: root.colAccentSecondary }
                            TextField {
                                id: createFolderField
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                placeholderText: "New folder..."
                                background: Item {}
                                color: root.colText
                                font.pixelSize: 12
                                enabled: root.backendInteractive
                                onAccepted: {
                                    if (text.trim().length === 0) return
                                    albumBackend.createFolder(text)
                                    text = ""
                                }
                            }
                            Rectangle {
                                width: 28; height: 28; radius: 6
                                color: addBtn.hovered ? root.colBorder : "transparent"
                                visible: root.backendInteractive && createFolderField.text.trim().length > 0
                                Label { anchors.centerIn: parent; text: "\u2713"; color: root.colAccentSecondary; font.pixelSize: 14; font.weight: 700 }
                                MouseArea {
                                    id: addBtn
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    property bool hovered: false
                                    onEntered: hovered = true
                                    onExited: hovered = false
                                    onClicked: {
                                        albumBackend.createFolder(createFolderField.text)
                                        createFolderField.text = ""
                                    }
                                }
                            }
                        }
                    }

                    // ── Delete-folder button ──
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 30
                        radius: 6
                        color: delBtn.hovered ? root.colDangerTint : "transparent"
                        border.width: 0
                        visible: root.backendInteractive && albumBackend.currentFolderId !== 0
                        Behavior on color { ColorAnimation { duration: 120 } }
                        Behavior on border.color { ColorAnimation { duration: 120 } }

                        RowLayout {
                            anchors.centerIn: parent
                            spacing: 6
                            Label { text: "\u{1F5D1}"; font.pixelSize: 12 }
                            Label { text: "Delete Folder"; color: root.colDanger; font.pixelSize: 12; font.weight: 600 }
                        }
                        MouseArea {
                            id: delBtn
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            property bool hovered: false
                            onEntered: hovered = true
                            onExited: hovered = false
                            onClicked: albumBackend.deleteFolder(albumBackend.currentFolderId)
                        }
                    }

                    // ── Separator ──
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 1
                        color: "#363636"
                        opacity: 1.0
                    }

                    // ── Folder card list ──
                    ListView {
                        id: folderList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 6
                        model: albumBackend.folders

                        delegate: Item {
                            required property int folderId
                            required property string name
                            required property int depth
                            required property string path
                            width: ListView.view.width
                            height: cardVisible ? cardHeight : 0
                            visible: cardVisible
                            Behavior on height { NumberAnimation { duration: 150; easing.type: Easing.OutCubic } }

                            readonly property bool cardVisible: folderSearchField.text.trim().length === 0
                                || name.toLowerCase().indexOf(folderSearchField.text.trim().toLowerCase()) >= 0
                            readonly property real cardHeight: 44 + depth * 0
                            readonly property bool isSelected: folderId === albumBackend.currentFolderId

                            Rectangle {
                                id: folderCard
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.leftMargin: depth * 12
                                height: parent.cardHeight
                                radius: 6
                                color: {
                                    if (isSelected) return root.colSelectedTint
                                    if (cardMouse.containsMouse) return root.colHover
                                    return "transparent"
                                }
                                border.width: 0
                                Behavior on color { ColorAnimation { duration: 140 } }

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    anchors.rightMargin: 10
                                    spacing: 8

                                    Label {
                                        text: isSelected ? "\u{1F4C2}" : "\u{1F4C1}"
                                        font.pixelSize: 16
                                    }

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 1
                                        Label {
                                            text: name
                                            Layout.fillWidth: true
                                            elide: Text.ElideRight
                                            color: isSelected ? root.colText : root.colText
                                            font.pixelSize: 13
                                            font.weight: isSelected ? 600 : 400
                                        }
                                        Label {
                                            visible: depth > 0
                                            text: path
                                            Layout.fillWidth: true
                                            elide: Text.ElideMiddle
                                            color: root.colTextMuted
                                            font.pixelSize: 10
                                        }
                                    }

                                    Rectangle {
                                        visible: isSelected
                                        width: 6; height: 6; radius: 3
                                        color: root.colAccentSecondary
                                    }
                                }

                                MouseArea {
                                    id: cardMouse
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    cursorShape: Qt.PointingHandCursor
                                    onClicked: {
                                        settingsPage = false
                                        albumBackend.selectFolder(folderId)
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: root.panelRadius
                color: root.colBgDeep
                clip: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 40
                    radius: 0
                    color: root.colBgDeep
                    border.width: 0
                    RowLayout {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.leftMargin: 14
                        anchors.rightMargin: 14
                        Label { text: "Browser"; color: root.colTextMuted; font.pixelSize: 13; font.weight: 600 }
                        Label { text: "Responsive thumbnail grid"; color: root.colTextMuted; font.pixelSize: 11 }
                        Item { Layout.fillWidth: true }
                        Button { text: "Grid"; checkable: true; checked: gridMode; onClicked: gridMode = true; flat: true }
                        Button { text: "List"; checkable: true; checked: !gridMode; onClicked: gridMode = false; flat: true }
                        Item { Layout.preferredWidth: 12 }
                        Button {
                            text: root.selectionMode ? "\u2611 Multi-Select" : "Multi-Select"
                            checkable: true
                            checked: root.selectionMode
                            onToggled: root.selectionMode = checked
                            flat: true
                            Material.foreground: root.selectionMode ? root.colAccentPrimary : root.colTextMuted
                        }
                    }
                }

                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: settingsPage ? 1 : 0

                    Rectangle {
                        radius: 0
                        color: root.colBgDeep
                        border.width: 0

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 14
                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: "Album"; color: root.colTextMuted; font.pixelSize: 14; font.weight: 600 }
                                Item { Layout.fillWidth: true }
                                Label { text: albumBackend.filterInfo; color: root.colTextMuted; font.pixelSize: 11 }
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
                                    color: root.colText
                                    font.pixelSize: 22
                                    font.weight: 700
                                }
                                Label {
                                    text: albumBackend.serviceReady
                                          ? "Import your first folder to start thumbnail generation and RAW adjustments."
                                          : "Use Load or New in the header to choose the database/metadata JSON files."
                                    color: root.colTextMuted
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
                        radius: 0
                        color: root.colBgDeep
                        border.width: 0
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 12
                            Label { text: "Settings"; color: root.colText; font.pixelSize: 20; font.weight: 700 }
                            Label { text: "Window #1A1A1A  Text #E6E6E6  Accent #FCC704"; color: root.colTextMuted; font.pixelSize: 12 }
                            Label { text: "Qt Quick renderer is hardware accelerated."; color: root.colTextMuted; font.pixelSize: 12 }
                            Item { Layout.fillHeight: true }
                        }
                    }
                }
            }

            } // close center block wrapper

            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: inspectorVisible && !settingsPage ? 350 : 0
                Behavior on Layout.preferredWidth { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }
                radius: root.panelRadius
                color: root.colBgPanel
                border.width: 0
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
            Layout.preferredHeight: 40
            radius: root.panelRadius
            color: root.colBgPanel
            border.width: 0

            Rectangle {
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                height: 2
                radius: 1
                color: albumBackend.projectLoading ? root.colAccentPrimary : "transparent"
            }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 10
                spacing: 12
                Label {
                    Layout.fillWidth: true
                    text: albumBackend.serviceMessage.length > 0 ? albumBackend.serviceMessage : albumBackend.taskStatus
                    elide: Text.ElideMiddle
                    color: albumBackend.projectLoading ? root.colAccentPrimary : root.colTextMuted
                    font.pixelSize: 11
                }
                ProgressBar { Layout.preferredWidth: 240; value: albumBackend.taskProgress / 100.0 }
                Button {
                    visible: albumBackend.taskCancelVisible
                    text: "Cancel"
                    Material.background: root.colDanger
                    Material.foreground: root.colText
                    onClicked: albumBackend.cancelImport()
                }
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
            color: root.colOverlay
        }

        MouseArea { anchors.fill: parent; hoverEnabled: true }

        Rectangle {
            anchors.centerIn: parent
            width: Math.min(parent.width - 36, 700)
            height: dialogContent.implicitHeight + 36
            radius: 14
            color: root.colBgDeep
            border.width: 0

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
                    color: root.colText
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: "Every boot asks for a project. Load a packed .puerhproj file or a metadata JSON/database pair, or create a new project package."
                    color: root.colTextMuted
                    font.pixelSize: 13
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: albumBackend.serviceMessage
                    color: root.colText
                    font.pixelSize: 12
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10

                    Button {
                        Layout.fillWidth: true
                        text: "Load Existing Project"
                        Material.background: root.colAccentPrimary
                        Material.foreground: root.colBgDeep
                        onClicked: loadProjectDialog.open()
                    }
                    Button {
                        Layout.fillWidth: true
                        text: "Create New Project"
                        Material.background: root.colAccentPrimary
                        Material.foreground: root.colBgDeep
                        onClicked: createProjectFolderDialog.open()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    Item { Layout.fillWidth: true }
                    Button {
                        text: "Exit"
                        Material.background: root.colDanger
                        Material.foreground: root.colText
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
            color: root.colOverlay
        }

        MouseArea { anchors.fill: parent; hoverEnabled: true }

        Rectangle {
            anchors.centerIn: parent
            width: Math.min(parent.width - 36, 520)
            height: loadingContent.implicitHeight + 30
            radius: 14
            color: root.colBgDeep
            border.width: 0

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
                    color: root.colText
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
                    color: root.colTextMuted
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
            selectionMode: root.selectionMode
            onImageSelectionChanged: function(elementId, imageId, fileName, selected) {
                selectionState.setImageSelected(elementId, imageId, fileName, selected)
            }
            onReplaceSelection: function(items) {
                selectionState.replaceSelectedImages(items)
            }
        }
    }

    Component {
        id: listComp
        ThumbnailListView {
            selectedImagesById: root.selectedImagesById
            exportQueueById: root.exportQueueById
            selectionMode: root.selectionMode
            onImageSelectionChanged: function(elementId, imageId, fileName, selected) {
                selectionState.setImageSelected(elementId, imageId, fileName, selected)
            }
            onReplaceSelection: function(items) {
                selectionState.replaceSelectedImages(items)
            }
        }
    }
}

