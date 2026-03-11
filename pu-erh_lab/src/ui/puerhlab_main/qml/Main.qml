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
    title: qsTr("Pu-erh Lab")
    font.family: appTheme.uiFontFamily

    // Theme palette — borderless, luminance-separated zones
    readonly property color toneGold: appTheme.toneGold
    readonly property color toneWine: appTheme.toneWine
    readonly property color toneSteel: appTheme.toneSteel
    readonly property color toneGraphite: appTheme.toneGraphite
    readonly property color toneMist: appTheme.toneMist
    readonly property color toneAmber: appTheme.toneGold
    readonly property color toneAccentSecondary: appTheme.accentSecondaryColor

    readonly property color colBgDeep: appTheme.bgDeepColor        // center content — darkest "stage"
    readonly property color colBgBase: appTheme.bgBaseColor        // sunken inputs
    readonly property color colBgPanel: appTheme.bgPanelColor      // side panels & header/footer
    readonly property color colBgCanvas: appTheme.bgCanvasColor    // gap / outer canvas behind blocks
    readonly property int panelRadius: appTheme.panelRadius        // uniform rounded-corner radius
    readonly property color colBorder: "transparent"     // NO borders by default
    readonly property color colText: appTheme.textColor
    readonly property color colTextMuted: appTheme.textMutedColor
    readonly property color colAccentPrimary: appTheme.accentColor
    readonly property color colAccentSecondary: appTheme.accentSecondaryColor
    readonly property color colAccentSoft: appTheme.accentColor
    readonly property color colDanger: appTheme.dangerColor
    readonly property color colDangerTint: appTheme.dangerTintColor
    readonly property color colSelectedTint: appTheme.selectedTintColor
    readonly property color colHover: appTheme.hoverColor          // subtle hover tint
    readonly property color colOverlay: appTheme.overlayColor
    readonly property string dataFontFamily: appTheme.dataFontFamily

    Material.theme: Material.Dark
    Material.primary: root.colAccentSecondary
    Material.accent: root.colAccentPrimary
    Material.background: root.colBgPanel
    Material.foreground: root.colText
    color: root.colBgPanel

    property bool settingsPage: false
    property bool inspectorVisible: true
    property real inspectorWidth: 350
    readonly property real inspectorMinWidth: 250
    readonly property real inspectorMaxWidth: 600
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
    readonly property var languageOptions: languageManager.availableLanguages
    property var pendingDeleteTargets: []
    property string deleteConfirmText: ""

    function languageIndexForCode(code) {
        for (let i = 0; i < languageOptions.length; ++i) {
            if (languageOptions[i].code === code) {
                return i
            }
        }
        return 0
    }

    function resolveDeleteTargets(clickedItem) {
        if (root.selectedCount > 0) {
            return Object.values(root.selectedImagesById)
        }
        if (!clickedItem) {
            return []
        }
        return [{
            elementId: Number(clickedItem.elementId),
            imageId: Number(clickedItem.imageId),
            fileName: clickedItem.fileName ? clickedItem.fileName : qsTr("(unnamed)")
        }]
    }

    function openDeleteContextMenu(clickedItem, sceneX, sceneY) {
        if (!root.backendInteractive) {
            return
        }
        const targets = resolveDeleteTargets(clickedItem)
        if (!targets || targets.length === 0) {
            return
        }
        root.pendingDeleteTargets = targets
        imageContextMenu.openAt(sceneX, sceneY)
    }

    function requestDeleteConfirmation() {
        const count = root.pendingDeleteTargets.length
        if (count <= 0) {
            return
        }
        if (count === 1) {
            root.deleteConfirmText = qsTr("Delete this image from project?")
        } else {
            root.deleteConfirmText = qsTr("Delete %1 images from project?").arg(count)
        }
        deleteConfirmDialog.open()
    }

    function runDeleteTargets() {
        if (!root.pendingDeleteTargets || root.pendingDeleteTargets.length === 0) {
            return
        }
        const result = albumBackend.DeleteImages(root.pendingDeleteTargets)
        const deletedIds = (result && result.deletedElementIds) ? result.deletedElementIds : []
        if (deletedIds.length > 0) {
            selectionState.pruneDeletedElements(deletedIds)
        }
        root.pendingDeleteTargets = []
    }

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
                    fileName: fileName ? fileName : qsTr("(unnamed)")
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
                    fileName: item.fileName ? item.fileName : qsTr("(unnamed)")
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

        function pruneDeletedElements(elementIds) {
            if (!elementIds || elementIds.length === 0) {
                return
            }

            const deleted = {}
            for (let i = 0; i < elementIds.length; ++i) {
                deleted[keyForElement(elementIds[i])] = true
            }

            const nextSelected = {}
            const selectedRows = Object.values(selectedImagesById)
            for (let i = 0; i < selectedRows.length; ++i) {
                const row = selectedRows[i]
                const key = keyForElement(row.elementId)
                if (!deleted[key]) {
                    nextSelected[key] = row
                }
            }
            selectedImagesById = nextSelected

            const nextQueue = {}
            const queueRows = Object.values(exportQueueById)
            for (let i = 0; i < queueRows.length; ++i) {
                const row = queueRows[i]
                const key = keyForElement(row.elementId)
                if (!deleted[key]) {
                    nextQueue[key] = row
                }
            }
            exportQueueById = nextQueue
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
                    label: qsTr("Image #%1  Sleeve #%2  %3")
                        .arg(item.imageId)
                        .arg(item.elementId)
                        .arg(item.fileName)
                })
            }
            if (src.length > previewCount) {
                next.push({ label: qsTr("... and %1 more").arg(src.length - previewCount) })
            }
            exportPreviewRows = next
        }
    }

    FileDialog {
        id: loadProjectDialog
        title: qsTr("Select Project Package or Metadata JSON")
        fileMode: FileDialog.OpenFile
        nameFilters: [
            qsTr("Packed Project (*.puerhproj)"),
            qsTr("Project Metadata (*.json)"),
            qsTr("All Files (*)")
        ]
        onAccepted: {
            albumBackend.LoadProject(selectedFile.toString())
        }
    }

    FolderDialog {
        id: createProjectFolderDialog
        title: qsTr("Select Parent Folder for New Project")
        onAccepted: {
            root.pendingNewProjectFolderUrl = selectedFolder.toString()
            createProjectNameField.text = root.defaultNewProjectName
            createProjectNameDialog.open()
        }
    }

    Dialog {
        id: createProjectNameDialog
        modal: true
        title: qsTr("Name New Project")
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
            albumBackend.CreateProjectInFolderNamed(root.pendingNewProjectFolderUrl, trimmed)
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
                text: qsTr("Choose the project package name. The app will create a single .puerhproj file.")
                wrapMode: Text.WordWrap
                color: root.colText
            }

            TextField {
                id: createProjectNameField
                Layout.fillWidth: true
                placeholderText: qsTr("Project name")
                onAccepted: createProjectNameDialog.submitCreateProject()
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                Item { Layout.fillWidth: true }

                Button {
                    text: qsTr("Cancel")
                    onClicked: createProjectNameDialog.close()
                }

                Button {
                    text: qsTr("Create")
                    enabled: createProjectNameField.text.trim().length > 0
                    onClicked: createProjectNameDialog.submitCreateProject()
                }
            }
        }
    }

    FileDialog {
        id: importDialog
        title: qsTr("Select Images")
        fileMode: FileDialog.OpenFiles
        nameFilters: [
            qsTr("Images (*.dng *.nef *.cr2 *.cr3 *.arw *.rw2 *.raf *.tif *.tiff *.jpg *.jpeg *.png)"),
            qsTr("All Files (*)")
        ]
        onAccepted: {
            const files = []
            for (let i = 0; i < selectedFiles.length; ++i) {
                files.push(selectedFiles[i].toString())
            }
            albumBackend.StartImport(files)
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
            albumBackend.StartExportWithOptionsForTargets(
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

    ImageContextMenu {
        id: imageContextMenu
        actions: [
            {
                id: "delete",
                label: qsTr("Delete"),
                enabled: root.pendingDeleteTargets.length > 0
            }
        ]
        onActionRequested: function(actionId) {
            if (actionId !== "delete") {
                return
            }
            imageContextMenu.close()
            requestDeleteConfirmation()
        }
    }

    Popup {
        id: deleteConfirmDialog
        modal: true
        focus: true
        closePolicy: Popup.CloseOnEscape
        width: Math.min(root.width - 36, 520)
        height: deleteConfirmContent.implicitHeight + 36
        x: Math.round((root.width - width) / 2)
        y: Math.round((root.height - height) / 2)

        Overlay.modal: Item {
            anchors.fill: parent

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
        }

        background: Rectangle {
            radius: 14
            color: root.colBgPanel
            border.width: 0
        }

        onClosed: {
            root.deleteConfirmText = ""
        }

        contentItem: ColumnLayout {
            id: deleteConfirmContent
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 18
            spacing: 12

            Label {
                text: qsTr("Confirm Deletion")
                font.pixelSize: 24
                font.weight: 700
                color: root.colText
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: root.deleteConfirmText.length > 0
                      ? qsTr("%1\nOriginal source files on disk will be kept.")
                            .arg(root.deleteConfirmText)
                      : ""
                color: root.colText
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                Item { Layout.fillWidth: true }

                Button {
                    text: qsTr("Cancel")
                    onClicked: {
                        root.pendingDeleteTargets = []
                        deleteConfirmDialog.close()
                    }
                }

                Button {
                    text: qsTr("Delete")
                    Material.background: root.colDanger
                    Material.foreground: root.colText
                    onClicked: {
                        deleteConfirmDialog.close()
                        root.runDeleteTargets()
                    }
                }
            }
        }
    }

    Connections {
        target: albumBackend
        ignoreUnknownSignals: true
        function onProjectChanged() {
            selectionState.clearSelectedImages()
            selectionState.clearExportQueue()
            root.pendingDeleteTargets = []
            deleteConfirmDialog.close()
            settingsPage = false
        }
        function onFolderSelectionChanged() {
            selectionState.clearSelectedImages()
            root.pendingDeleteTargets = []
            deleteConfirmDialog.close()
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
                Label { text: qsTr("Pu-erh Lab"); font.pixelSize: 19; font.weight: 700; color: root.colAccentPrimary }
                Item { Layout.preferredWidth: 8 }

                // ── Load / New / Save pill ──
                Rectangle {
                    id: projectPill
                    Layout.preferredHeight: 36
                    Layout.preferredWidth: pillRow.implicitWidth + 8
                    radius: 6
                    color: root.colBgBase
                    border.width: 0

                    Row {
                        id: pillRow
                        anchors.centerIn: parent
                        spacing: 0

                        Repeater {
                            model: [
                                { label: qsTr("Load"), act: "load",  en: !albumBackend.projectLoading },
                                { label: qsTr("New"),  act: "new",   en: !albumBackend.projectLoading },
                                { label: qsTr("Save"), act: "save",  en: root.backendInteractive }
                            ]
                            delegate: Item {
                                width: pillSegment.width + (index < 2 ? pillDivider.width : 0)
                                height: projectPill.height

                                Rectangle {
                                    id: pillSegment
                                    width: pillLabel.implicitWidth + 28
                                    height: parent.height - 4
                                    anchors.verticalCenter: parent.verticalCenter
                                    radius: 4
                                    color: pillMouse.containsMouse && modelData.en
                                           ? root.colHover : "transparent"
                                    Behavior on color { ColorAnimation { duration: 120 } }

                                    Label {
                                        id: pillLabel
                                        anchors.centerIn: parent
                                        text: modelData.label
                                        font.pixelSize: 12
                                        font.weight: 500
                                        color: modelData.en ? root.colText : root.colTextMuted
                                        opacity: modelData.en ? 1.0 : 0.45
                                    }

                                    MouseArea {
                                        id: pillMouse
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: modelData.en ? Qt.PointingHandCursor : Qt.ArrowCursor
                                        onClicked: {
                                            if (!modelData.en) return
                                            if (modelData.act === "load")
                                                loadProjectDialog.open()
                                            else if (modelData.act === "new")
                                                createProjectFolderDialog.open()
                                            else if (modelData.act === "save")
                                                albumBackend.SaveProject()
                                        }
                                    }
                                }

                                // thin divider between segments
                                Rectangle {
                                    id: pillDivider
                                    visible: index < 2
                                    anchors.right: parent.right
                                    anchors.verticalCenter: parent.verticalCenter
                                    width: 1
                                    height: parent.height * 0.48
                                    color: root.toneSteel
                                }
                            }
                        }
                    }
                }

                Item { Layout.preferredWidth: 4 }
                Button {
                    text: qsTr("Import")
                    enabled: root.backendInteractive
                    height: 36
                    Material.background: root.colAccentSoft
                    Material.foreground: root.colBgDeep
                    onClicked: importDialog.open()
                }
                Item { Layout.fillWidth: true }
                Button {
                    text: qsTr("Add Selected (%1)").arg(root.selectedCount)
                    enabled: root.backendInteractive && root.selectedCount > 0
                    Material.background: root.colAccentPrimary
                    Material.foreground: root.colBgDeep
                    onClicked: selectionState.addSelectedToExportQueue()
                }
                Button {
                    text: qsTr("Export Queue (%1)").arg(root.exportQueueCount)
                    enabled: root.backendInteractive && (albumBackend.shownCount > 0 || root.exportQueueCount > 0)
                    Material.background: root.colAccentSecondary
                    Material.foreground: root.colBgDeep
                    onClicked: {
                        selectionState.refreshExportPreview()
                        exportDialog.open()
                    }
                }
                Item { Layout.preferredWidth: 8 }
                Button { text: qsTr("Library"); checkable: true; checked: !settingsPage; onClicked: settingsPage = false }
                Button { text: qsTr("Settings"); checkable: true; checked: settingsPage; onClicked: settingsPage = true }
                Button { text: qsTr("Inspector"); checkable: true; checked: inspectorVisible; onToggled: inspectorVisible = checked }
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
                            text: qsTr("Library")
                            font.pixelSize: 17
                            font.weight: 700
                            color: root.colText
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: qsTr("%1 folders").arg(folderList.count)
                            color: root.colAccentSoft
                            font.family: root.dataFontFamily
                            font.pixelSize: 11
                        }
                    }

                    Label {
                        text: albumBackend.currentFolderPath
                        color: root.colTextMuted
                        font.family: root.dataFontFamily
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
                                placeholderText: qsTr("Search folders...")
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
                                placeholderText: qsTr("New folder...")
                                background: Item {}
                                color: root.colText
                                font.pixelSize: 12
                                enabled: root.backendInteractive
                                onAccepted: {
                                    if (text.trim().length === 0) return
                                    albumBackend.CreateFolder(text)
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
                                        albumBackend.CreateFolder(createFolderField.text)
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
                        color: root.colDanger
                        border.width: 0
                        visible: root.backendInteractive && albumBackend.currentFolderId !== 0
                        Behavior on color { ColorAnimation { duration: 120 } }
                        Behavior on border.color { ColorAnimation { duration: 120 } }

                        RowLayout {
                            anchors.centerIn: parent
                            spacing: 6
                            Label { text: "\u{1F5D1}"; font.pixelSize: 12 }
                            Label { text: qsTr("Delete Folder"); font.pixelSize: 12; font.weight: 600 }
                        }
                        MouseArea {
                            id: delBtn
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            property bool hovered: false
                            onEntered: hovered = true
                            onExited: hovered = false
                            onClicked: albumBackend.DeleteFolder(albumBackend.currentFolderId)
                        }
                    }

                    // ── Separator ──
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 1
                        color: root.toneSteel
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
                                        albumBackend.SelectFolder(folderId)
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
                        Label { text: qsTr("Browser"); color: root.colTextMuted; font.pixelSize: 13; font.weight: 600 }
                        Item { Layout.fillWidth: true }
                        Button { text: qsTr("Grid"); checkable: true; checked: gridMode; onClicked: gridMode = true; flat: true }
                        Button { text: qsTr("List"); checkable: true; checked: !gridMode; onClicked: gridMode = false; flat: true }
                        Item { Layout.preferredWidth: 12 }
                        Button {
                            text: root.selectionMode ? qsTr("\u2611 Multi-Select") : qsTr("Multi-Select")
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
                                Label { text: qsTr("Album"); color: root.colTextMuted; font.pixelSize: 14; font.weight: 600 }
                                Item { Layout.fillWidth: true }
                                Label { text: albumBackend.filterInfo; color: root.colTextMuted; font.family: root.dataFontFamily; font.pixelSize: 11 }
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
                                    text: albumBackend.serviceReady ? qsTr("No Photos Yet") : qsTr("Open or Create a Project")
                                    color: root.colText
                                    font.pixelSize: 22
                                    font.weight: 700
                                }
                                Label {
                                    text: albumBackend.serviceReady
                                          ? qsTr("Import your images for RAW adjustments.")
                                          : qsTr("Use Load or New in the header to choose the .puerhproj files.")
                                    color: root.colTextMuted
                                    font.pixelSize: 12
                                }
                                Button {
                                    text: albumBackend.serviceReady ? qsTr("Import Photos") : qsTr("Load Project")
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
                            Label { text: qsTr("Settings"); color: root.colText; font.pixelSize: 20; font.weight: 700 }
                            Label { text: qsTr("Theme tokens: Window #1A1A1A  Text #E6E6E6  Accent #FCC704"); color: root.colTextMuted; font.pixelSize: 12 }
                            Label { text: qsTr("Qt Quick renderer is hardware accelerated."); color: root.colTextMuted; font.pixelSize: 12 }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label {
                                    text: qsTr("Language")
                                    color: root.colText
                                    font.pixelSize: 13
                                    font.weight: 600
                                }
                                ComboBox {
                                    Layout.preferredWidth: 220
                                    model: root.languageOptions
                                    textRole: "label"
                                    currentIndex: root.languageIndexForCode(languageManager.currentLanguageCode)
                                    onActivated: function(index) {
                                        const item = model[index]
                                        if (item) {
                                            languageManager.setLanguage(item.code)
                                        }
                                    }
                                }
                            }
                            Item { Layout.fillHeight: true }
                        }
                    }
                }
            }

            } // close center block wrapper

            // ── drag handle to resize inspector panel ──
            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: inspectorVisible && !settingsPage ? 5 : 0
                color: dragArea.containsMouse || dragArea.drag.active ? root.colAccentPrimary : "transparent"
                visible: inspectorVisible && !settingsPage
                Behavior on color { ColorAnimation { duration: 120 } }

                MouseArea {
                    id: dragArea
                    anchors.fill: parent
                    anchors.margins: -3          // widen the hit area
                    hoverEnabled: true
                    cursorShape: Qt.SplitHCursor
                    property real startX: 0
                    property real startWidth: 0
                    onPressed: function(mouse) {
                        startX = mapToGlobal(mouse.x, 0).x
                        startWidth = root.inspectorWidth
                    }
                    onPositionChanged: function(mouse) {
                        if (!pressed) return
                        var globalX = mapToGlobal(mouse.x, 0).x
                        var delta = startX - globalX   // dragging left ⇒ wider
                        root.inspectorWidth = Math.max(root.inspectorMinWidth,
                                                      Math.min(root.inspectorMaxWidth,
                                                               startWidth + delta))
                    }
                }
            }

            Rectangle {
                Layout.fillHeight: true
                Layout.preferredWidth: inspectorVisible && !settingsPage ? root.inspectorWidth : 0
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
                    text: qsTr("Cancel")
                    Material.background: root.colDanger
                    Material.foreground: root.colText
                    onClicked: albumBackend.CancelImport()
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
            color: root.colBgPanel
            border.width: 0

            ColumnLayout {
                id: dialogContent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 18
                spacing: 12

                Label {
                    text: qsTr("Open Project")
                    font.pixelSize: 24
                    font.weight: 700
                    color: root.colText
                }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: qsTr("Every boot asks for a project. Load a packed .puerhproj file or a metadata JSON/database pair, or create a new project package.")
                    color: root.colTextMuted
                    font.pixelSize: 13
                }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label {
                        text: qsTr("Language")
                        color: root.colText
                        font.pixelSize: 12
                        font.weight: 600
                    }
                    ComboBox {
                        Layout.preferredWidth: 220
                        model: root.languageOptions
                        textRole: "label"
                        currentIndex: root.languageIndexForCode(languageManager.currentLanguageCode)
                        onActivated: function(index) {
                            const item = model[index]
                            if (item) {
                                languageManager.setLanguage(item.code)
                            }
                        }
                    }
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
                        text: qsTr("Load Existing Project")
                        Material.background: root.colAccentPrimary
                        Material.foreground: root.colBgDeep
                        onClicked: loadProjectDialog.open()
                    }
                    Button {
                        Layout.fillWidth: true
                        text: qsTr("Create New Project")
                        Material.background: root.colAccentPrimary
                        Material.foreground: root.colBgDeep
                        onClicked: createProjectFolderDialog.open()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    Item { Layout.fillWidth: true }
                    Button {
                        text: qsTr("Exit")
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
                    text: qsTr("Loading Project")
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
                          : qsTr("Preparing database and metadata...")
                    color: root.colTextMuted
                    font.pixelSize: 12
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }
    }

    // ── Import progress overlay ──────────────────────────────────────────
    Item {
        anchors.fill: parent
        visible: albumBackend.importRunning
        z: 50

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
            width: Math.min(parent.width - 36, 420)
            height: importDialogContent.implicitHeight + 36
            radius: 14
            color: root.colBgDeep
            border.width: 0

            ColumnLayout {
                id: importDialogContent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 20
                spacing: 16

                Label {
                    text: qsTr("Importing Photos")
                    font.pixelSize: 21
                    font.weight: 700
                    color: root.colText
                    Layout.alignment: Qt.AlignHCenter
                }

                ImportProgressRing {
                    Layout.alignment: Qt.AlignHCenter
                    width: 160
                    height: 160
                    ringWidth: 14
                    trackColor: root.colHover
                    fillColor: root.colAccentPrimary
                    progress: albumBackend.importTotal > 0
                              ? albumBackend.importCompleted / albumBackend.importTotal
                              : 0
                }

                Label {
                    Layout.alignment: Qt.AlignHCenter
                    text: qsTr("%1 / %2").arg(albumBackend.importCompleted).arg(albumBackend.importTotal)
                    font.family: root.dataFontFamily
                    font.pixelSize: 28
                    font.weight: 600
                    color: root.colText
                }

                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    horizontalAlignment: Text.AlignHCenter
                    text: albumBackend.importStatus.length > 0
                          ? albumBackend.importStatus
                          : qsTr("Preparing...")
                    color: root.colTextMuted
                    font.pixelSize: 12
                }

                Label {
                    Layout.fillWidth: true
                    visible: albumBackend.importFailed > 0
                    wrapMode: Text.WordWrap
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("%1 file(s) failed").arg(albumBackend.importFailed)
                    color: root.colDanger
                    font.family: root.dataFontFamily
                    font.pixelSize: 12
                }

                Button {
                    Layout.alignment: Qt.AlignHCenter
                    text: qsTr("Cancel")
                    Material.background: root.colDanger
                    Material.foreground: root.colText
                    onClicked: albumBackend.CancelImport()
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
            onContextMenuRequested: function(item, sceneX, sceneY) {
                root.openDeleteContextMenu(item, sceneX, sceneY)
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
            onContextMenuRequested: function(item, sceneX, sceneY) {
                root.openDeleteContextMenu(item, sceneX, sceneY)
            }
        }
    }
}
