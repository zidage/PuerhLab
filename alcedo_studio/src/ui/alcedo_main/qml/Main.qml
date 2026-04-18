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
    title: qsTr("Alcedo Studio")
    font.family: appTheme.uiFontFamily

    // Theme palette — borderless, luminance-separated zones
    readonly property color toneGold: appTheme.toneGold
    readonly property color toneWine: appTheme.toneWine
    readonly property color toneSteel: appTheme.toneSteel
    readonly property color toneGraphite: appTheme.toneGraphite
    readonly property color toneMist: appTheme.toneMist
    readonly property color toneAmber: appTheme.toneGold
    readonly property color toneAccentSecondary: appTheme.accentSecondaryColor

    readonly property color colBgDeep: appTheme.bgDeepColor        // floating modals / popovers — topmost layer
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
    readonly property color colDivider: appTheme.dividerColor
    readonly property color colGlassPanel: appTheme.glassPanelColor
    readonly property color colGlassStroke: appTheme.glassStrokeColor
    readonly property color colOverlay: appTheme.overlayColor
    readonly property string dataFontFamily: appTheme.dataFontFamily
    readonly property string headlineFontFamily: appTheme.headlineFontFamily
    readonly property real settingsFieldLabelWidth: 84
    readonly property int controlRadius: 10
    readonly property color colButtonPrimary: "#457B9D"
    readonly property color colButtonSecondary: "#3A3F44"
    readonly property color colButtonHighlight: "#E9C46A"
    readonly property color colButtonBorder: Qt.rgba(
        colButtonHighlight.r,
        colButtonHighlight.g,
        colButtonHighlight.b,
        0.20)
    readonly property color colButtonSecondaryBorder: Qt.rgba(
        root.colText.r,
        root.colText.g,
        root.colText.b,
        0.12)

    function withAlpha(colorValue, alphaValue) {
        return Qt.rgba(colorValue.r, colorValue.g, colorValue.b, alphaValue)
    }

    function nauticalButtonFill(enabled, hovered, pressed) {
        if (!enabled) {
            return withAlpha(colButtonPrimary, 0.45)
        }
        if (pressed) {
            return Qt.darker(colButtonPrimary, 1.18)
        }
        if (hovered) {
            return Qt.lighter(colButtonPrimary, 1.08)
        }
        return colButtonPrimary
    }

    function secondaryButtonFill(enabled, hovered, pressed) {
        if (!enabled) {
            return withAlpha(colButtonSecondary, 0.55)
        }
        if (pressed) {
            return Qt.darker(colButtonSecondary, 1.14)
        }
        if (hovered) {
            return Qt.lighter(colButtonSecondary, 1.08)
        }
        return colButtonSecondary
    }

    Material.theme: Material.Dark
    Material.primary: root.colAccentSecondary
    Material.accent: root.colAccentPrimary
    Material.background: root.colBgPanel
    Material.foreground: root.colText
    color: root.colBgPanel

    property bool inspectorVisible: true
    property real inspectorWidth: 300
    readonly property real inspectorMinWidth: 300
    readonly property real inspectorMaxWidth: 600
    readonly property real leftPaneWidth: 250
    readonly property real centerPaneMinWidth: 560
    readonly property real mainFrameHorizontalMargins: 24
    readonly property real contentRowSpacingTotal: 36
    readonly property real inspectorAdaptiveMaxWidth: Math.max(
        0,
        root.width
        - leftPaneWidth
        - centerPaneMinWidth
        - mainFrameHorizontalMargins
        - contentRowSpacingTotal
        - 5)
    property bool gridMode: true
    readonly property bool backendInteractive: albumBackend.serviceReady && !albumBackend.projectLoading
    readonly property var selectedImagesById: selectionState.selectedImagesById
    readonly property var exportQueueById: selectionState.exportQueueById
    readonly property var exportPreviewRows: selectionState.exportPreviewRows
    readonly property int selectedCount: selectionState.selectedCount
    readonly property int exportQueueCount: selectionState.exportQueueCount
    readonly property var languageOptions: languageManager.availableLanguages
    property int pendingThemeIndex: appTheme.currentThemeIndex
    property string pendingLanguageCode: languageManager.currentLanguageCode
    property var pendingDeleteTargets: []
    property var pendingDetailsTarget: ({})
    property string deleteConfirmText: ""
    property string snackbarText: ""
    property bool importSessionObserved: false
    property bool exportSessionObserved: false
    property bool welcomeDismissedForLaunch: false
    property var imageDetailsData: ({
        title: "",
        subtitle: "",
        rows: []
    })

    onWelcomeDismissedForLaunchChanged: updateWelcomeDialogVisibility()
    Component.onCompleted: updateWelcomeDialogVisibility()

    function showSnackbar(messageText) {
        if (!messageText || String(messageText).trim().length === 0) {
            return
        }
        root.snackbarText = String(messageText)
        snackbarTimer.restart()
        if (!notificationSnackbar.opened) {
            notificationSnackbar.open()
        }
    }

    function requestSaveProject() {
        const ok = albumBackend.SaveProject()
        if (ok) {
            showSnackbar(albumBackend.serviceMessage)
        }
    }

    function languageIndexForCode(code) {
        for (let i = 0; i < languageOptions.length; ++i) {
            if (languageOptions[i].code === code) {
                return i
            }
        }
        return 0
    }

    function themeModelIndexForTheme(themeIndex) {
        const themes = appTheme.availableThemes
        for (let i = 0; i < themes.length; ++i) {
            if (themes[i].index === themeIndex) {
                return i
            }
        }
        return 0
    }

    function openSettingsDialog() {
        root.pendingThemeIndex = appTheme.currentThemeIndex
        root.pendingLanguageCode = languageManager.currentLanguageCode
        settingsDialog.open()
    }

    function saveSettingsAndClose() {
        if (appTheme.currentThemeIndex !== root.pendingThemeIndex) {
            appTheme.currentThemeIndex = root.pendingThemeIndex
        }
        if (languageManager.currentLanguageCode !== root.pendingLanguageCode) {
            languageManager.setLanguage(root.pendingLanguageCode)
        }
        settingsDialog.close()
    }

    function dismissWelcomeForProjectLaunch() {
        root.welcomeDismissedForLaunch = true
    }

    function updateWelcomeDialogVisibility() {
        const shouldShowWelcome = !root.welcomeDismissedForLaunch
                                  && !albumBackend.serviceReady
                                  && !albumBackend.projectLoading
        if (shouldShowWelcome) {
            if (!welcomeDialog.opened) {
                welcomeDialog.open()
            }
        } else if (welcomeDialog.opened) {
            welcomeDialog.close()
        }
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

    function openImageContextMenu(clickedItem, sceneX, sceneY) {
        if (!root.backendInteractive) {
            return
        }
        if (!clickedItem) {
            return
        }
        const targets = resolveDeleteTargets(clickedItem)
        if (!targets || targets.length === 0) {
            return
        }
        root.pendingDeleteTargets = targets
        root.pendingDetailsTarget = {
            elementId: Number(clickedItem.elementId),
            imageId: Number(clickedItem.imageId),
            fileName: clickedItem.fileName ? clickedItem.fileName : qsTr("(unnamed)")
        }
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

    function requestImageDetails() {
        if (!root.pendingDetailsTarget || Number(root.pendingDetailsTarget.imageId) <= 0) {
            return
        }
        const result = albumBackend.GetImageDetails(
            Number(root.pendingDetailsTarget.elementId),
            Number(root.pendingDetailsTarget.imageId))
        if (!result || result.success !== true) {
            imageDetailsDialog.close()
            return
        }
        root.imageDetailsData = {
            title: result.title ? result.title : qsTr("(unnamed)"),
            subtitle: result.subtitle ? result.subtitle : "",
            rows: result.rows ? result.rows : []
        }
        imageDetailsDialog.open()
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
        id: importDialog
        title: qsTr("Select Images")
        fileMode: FileDialog.OpenFiles
        nameFilters: [
            qsTr("RAW Images (*.raw *.dng *.nef *.cr2 *.cr3 *.arw *.rw2 *.raf *.3fr *.fff)"),
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
        hdrExportAvailable: root.exportQueueCount > 0
            && albumBackend.CanUseHdrExportForTargets(Object.values(root.exportQueueById))
        onAddSelectedToQueueRequested: selectionState.addSelectedToExportQueue()
        onClearQueueRequested: selectionState.clearExportQueue()
        onEnsurePreviewRequested: selectionState.refreshExportPreview()
        onStartExportRequested: function(outDir, format, hdrExportMode, resizeEnabled, maxSide, quality, bitDepth, pngLevel, tiffComp) {
            albumBackend.StartExportWithOptionsForTargets(
                outDir,
                format,
                hdrExportMode,
                resizeEnabled,
                maxSide,
                quality,
                bitDepth,
                pngLevel,
                tiffComp,
                selectionState.exportQueueTargets())
        }
    }

    Popup {
        id: settingsDialog
        modal: true
        focus: true
        closePolicy: Popup.CloseOnEscape
        width: Math.min(root.width - 36, 520)
        height: settingsDialogContent.implicitHeight + 36
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

        contentItem: ColumnLayout {
            id: settingsDialogContent
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.margins: 18
            spacing: 12

            Label {
                text: qsTr("Settings")
                font.family: root.headlineFontFamily
                font.pixelSize: 24
                font.weight: 700
                color: root.colText
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: qsTr("Choose theme and language, then save to apply changes.")
                color: root.colTextMuted
                font.pixelSize: 12
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10
                Label {
                    Layout.preferredWidth: root.settingsFieldLabelWidth
                    text: qsTr("Theme")
                    color: root.colText
                    font.pixelSize: 13
                    font.weight: 600
                }
                ComboBox {
                    Layout.fillWidth: true
                    model: appTheme.availableThemes
                    textRole: "label"
                    currentIndex: root.themeModelIndexForTheme(root.pendingThemeIndex)
                    onActivated: function(index) {
                        const item = model[index]
                        if (item) {
                            root.pendingThemeIndex = item.index
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10
                Label {
                    Layout.preferredWidth: root.settingsFieldLabelWidth
                    text: qsTr("Language")
                    color: root.colText
                    font.pixelSize: 13
                    font.weight: 600
                }
                ComboBox {
                    Layout.fillWidth: true
                    model: root.languageOptions
                    textRole: "label"
                    currentIndex: root.languageIndexForCode(root.pendingLanguageCode)
                    onActivated: function(index) {
                        const item = model[index]
                        if (item) {
                            root.pendingLanguageCode = item.code
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                Item { Layout.fillWidth: true }

                Button {
                    id: settingsCloseButton
                    text: qsTr("Close")
                    Material.background: root.colDanger
                    Material.foreground: root.colText
                    onClicked: settingsDialog.close()
                }

                Button {
                    id: settingsSaveButton
                    text: qsTr("Save")
                    Material.background: root.colButtonPrimary
                    Material.foreground: root.colText
                    onClicked: root.saveSettingsAndClose()
                }
            }
        }
    }

    ImageContextMenu {
        id: imageContextMenu
        actions: [
            {
                id: "details",
                label: qsTr("Details"),
                enabled: Number(root.pendingDetailsTarget.imageId) > 0
            },
            {
                id: "delete",
                label: qsTr("Delete"),
                enabled: root.pendingDeleteTargets.length > 0
            }
        ]
        onActionRequested: function(actionId) {
            imageContextMenu.close()
            if (actionId === "details") {
                requestImageDetails()
                return
            }
            if (actionId === "delete") {
                requestDeleteConfirmation()
            }
        }
    }

    ImageDetailsDialog {
        id: imageDetailsDialog
        parent: Overlay.overlay
        titleText: root.imageDetailsData.title
        subtitleText: root.imageDetailsData.subtitle
        detailRows: root.imageDetailsData.rows
        onRowActionRequested: function(actionId, actionValue) {
            if (actionId === "open-directory") {
                albumBackend.OpenDirectoryInFileManager(actionValue)
            }
        }
        onClosed: {
            root.imageDetailsData = {
                title: "",
                subtitle: "",
                rows: []
            }
        }
    }

    NikonHeRecoveryDialog {
        id: nikonHeRecoveryDialog
        parent: Overlay.overlay
        backgroundSource: mainContent
        recoveryActive: albumBackend.nikonHeRecoveryActive
        recoveryBusy: albumBackend.nikonHeRecoveryBusy
        recoveryPhase: albumBackend.nikonHeRecoveryPhase
        recoveryStatus: albumBackend.nikonHeRecoveryStatus
        unsupportedFiles: albumBackend.nikonHeUnsupportedFiles
        converterPath: albumBackend.nikonHeConverterPath
        showImportProgress: albumBackend.importRunning && albumBackend.nikonHeRecoveryActive
        importCompleted: albumBackend.importCompleted
        importTotal: albumBackend.importTotal
        importFailed: albumBackend.importFailed
        onBrowseRequested: albumBackend.BrowseNikonHeConverter()
        onConvertRequested: albumBackend.StartNikonHeConversion()
        onExitRequested: albumBackend.ExitNikonHeRecovery()
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
                font.family: root.headlineFontFamily
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
                    id: deleteCancelButton
                    text: qsTr("Cancel")
                    Material.background: root.colButtonSecondary
                    Material.foreground: root.colText
                    onClicked: {
                        root.pendingDeleteTargets = []
                        deleteConfirmDialog.close()
                    }
                }

                Button {
                    id: deleteConfirmButton
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
            root.welcomeDismissedForLaunch = false
            root.updateWelcomeDialogVisibility()
            selectionState.clearSelectedImages()
            selectionState.clearExportQueue()
            root.pendingDeleteTargets = []
            root.pendingDetailsTarget = ({})
            deleteConfirmDialog.close()
            imageDetailsDialog.close()
            root.showSnackbar(albumBackend.serviceMessage)
        }
        function onFolderSelectionChanged() {
            selectionState.clearSelectedImages()
            root.pendingDeleteTargets = []
            root.pendingDetailsTarget = ({})
            deleteConfirmDialog.close()
            imageDetailsDialog.close()
        }
        function onServiceStateChanged() {
            root.updateWelcomeDialogVisibility()
        }
        function onProjectLoadStateChanged() {
            if (!albumBackend.projectLoading) {
                root.welcomeDismissedForLaunch = false
            }
            root.updateWelcomeDialogVisibility()
        }
        function onThumbnailsChanged() {
            if (exportDialog.visible) {
                selectionState.refreshExportPreview()
            }
        }
        function onImportStateChanged() {
            if (albumBackend.importRunning) {
                root.importSessionObserved = true
                return
            }
            if (!root.importSessionObserved) {
                return
            }
            root.importSessionObserved = false
            root.showSnackbar(qsTr("Imported %1 image(s).").arg(albumBackend.importCompleted))
        }
        function onExportStateChanged() {
            if (albumBackend.exportInFlight) {
                root.exportSessionObserved = true
                return
            }
            if (!root.exportSessionObserved) {
                return
            }
            root.exportSessionObserved = false
            root.showSnackbar(qsTr("Exported %1 image(s).").arg(albumBackend.exportSucceeded))
        }
    }

    Popup {
        id: notificationSnackbar
        parent: Overlay.overlay
        modal: false
        focus: false
        closePolicy: Popup.NoAutoClose
        padding: 12
        width: Math.min(root.width - 24, 760)
        x: Math.round((root.width - width) / 2)
        y: root.height - height - 16

        background: Rectangle {
            radius: 10
            color: root.colGlassPanel
            border.width: 1
            border.color: root.colGlassStroke
        }

        contentItem: Label {
            text: root.snackbarText
            color: root.colText
            wrapMode: Text.WordWrap
            horizontalAlignment: Text.AlignHCenter
        }

        enter: Transition {
            NumberAnimation { property: "opacity"; from: 0.0; to: 1.0; duration: 120 }
        }
        exit: Transition {
            NumberAnimation { property: "opacity"; from: 1.0; to: 0.0; duration: 120 }
        }
    }

    Timer {
        id: snackbarTimer
        interval: 2600
        repeat: false
        onTriggered: notificationSnackbar.close()
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
        anchors.margins: 12
        spacing: 12

        Rectangle {
            id: topToolbar
            Layout.fillWidth: true
            Layout.preferredHeight: 56
            radius: root.panelRadius
            color: root.colGlassPanel
            border.width: 1
            border.color: root.colGlassStroke
            z: 1

            RowLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 10
                Row {
                    spacing: 0
                    Label { text: qsTr("Alcedo"); font.family: root.headlineFontFamily; font.pixelSize: 19; font.weight: 700; color: root.colAccentPrimary }
                    Label { text: " "; font.family: root.headlineFontFamily; font.pixelSize: 19; font.weight: 700 }
                    Label { text: qsTr("Lab"); font.family: root.headlineFontFamily; font.pixelSize: 19; font.weight: 700; color: root.colText }
                }
                Item { Layout.preferredWidth: 12 }

                // ── File menu ──
                Button {
                    id: fileMenuButton
                    text: qsTr("File")
                    flat: true
                    Material.foreground: root.colText
                    onClicked: fileMenu.open()

                    Menu {
                        id: fileMenu
                        x: 0
                        y: fileMenuButton.height + 4

                        MenuItem {
                            text: qsTr("Load Project")
                            enabled: !albumBackend.projectLoading
                            onTriggered: albumBackend.PromptAndLoadProject()
                        }
                        MenuItem {
                            text: qsTr("Create Project")
                            enabled: !albumBackend.projectLoading
                            onTriggered: albumBackend.PromptAndCreateProject()
                        }
                        MenuSeparator {
                        }
                        MenuItem {
                            text: qsTr("Save Project")
                            enabled: root.backendInteractive
                            onTriggered: root.requestSaveProject()
                        }
                    }
                }

                Button {
                    id: settingsPopoutButton
                    text: qsTr("Settings")
                    flat: true
                    Material.foreground: root.colText
                    onClicked: root.openSettingsDialog()
                }

                Item { Layout.fillWidth: true }
                Button {
                    id: inspectorToggleButton
                    checkable: false
                    flat: true
                    Layout.preferredWidth: 52
                    Layout.preferredHeight: 42
                    display: AbstractButton.IconOnly
                    property real iconRotationTarget: inspectorVisible ? 180 : 0
                    icon.source: "qrc:/panel_icons/inspector-expand.svg"
                    icon.width: 24
                    icon.height: 24
                    icon.color: inspectorVisible
                                ? root.colAccentPrimary
                                : (inspectorToggleButton.hovered ? root.colText : root.colTextMuted)
                    Material.foreground: icon.color
                    ToolTip.visible: hovered
                    ToolTip.text: inspectorVisible ? qsTr("Collapse Inspector") : qsTr("Expand Inspector")
                    background: Rectangle {
                        radius: root.controlRadius
                        color: "transparent"
                        border.width: 0
                    }
                    onContentItemChanged: {
                        inspectorIconRotate.target = contentItem
                        if (contentItem) {
                            contentItem.transformOrigin = Item.Center
                            contentItem.rotation = iconRotationTarget
                        }
                    }
                    onIconRotationTargetChanged: {
                        if (contentItem) {
                            inspectorIconRotate.stop()
                            inspectorIconRotate.to = iconRotationTarget
                            inspectorIconRotate.start()
                        }
                    }
                    Component.onCompleted: {
                        if (contentItem) {
                            contentItem.transformOrigin = Item.Center
                            contentItem.rotation = iconRotationTarget
                        }
                    }
                    NumberAnimation {
                        id: inspectorIconRotate
                        property: "rotation"
                        duration: 170
                        easing.type: Easing.OutCubic
                    }
                    onClicked: inspectorVisible = !inspectorVisible
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            ColumnLayout {
                Layout.preferredWidth: 250
                Layout.minimumWidth: 250
                Layout.maximumWidth: 250
                Layout.fillHeight: true
                spacing: 10

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: root.panelRadius
                    color: root.colGlassPanel
                    border.width: 1
                    border.color: root.colGlassStroke
                    clip: true

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 12

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
                                font.family: appTheme.uiFontFamily
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
                            Layout.preferredHeight: 38
                            radius: root.controlRadius
                            color: root.colBgBase
                            border.width: 1
                            border.color: root.colDivider
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
                            Layout.preferredHeight: 38
                            radius: root.controlRadius
                            color: root.colBgBase
                            border.width: 1
                            border.color: root.colDivider
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
                                    width: 28; height: 28; radius: 8
                                    color: addBtn.hovered ? root.colHover : "transparent"
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
                            radius: root.controlRadius
                            color: root.colDanger
                            border.width: 1
                            border.color: Qt.rgba(root.colDanger.r, root.colDanger.g, root.colDanger.b, 0.24)
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
                            color: root.colDivider
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
                                    radius: 8
                                    color: {
                                        if (isSelected) return root.colSelectedTint
                                        if (cardMouse.containsMouse) return root.colHover
                                        return "transparent"
                                    }
                                    border.width: isSelected ? 1 : 0
                                    border.color: root.colGlassStroke
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
                                            albumBackend.SelectFolder(folderId)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Button {
                    id: importBtn
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    text: qsTr("Import")
                    enabled: root.backendInteractive
                    icon.source: "qrc:/panel_icons/import.svg"
                    icon.width: 16
                    icon.height: 16
                    icon.color: root.colText
                    display: AbstractButton.TextBesideIcon
                    background: Canvas {
                        opacity: importBtn.enabled ? 1.0 : 0.5
                        property color gradStart: root.colAccentPrimary
                        property color gradEnd: root.colAccentSecondary
                        onGradStartChanged: requestPaint()
                        onGradEndChanged: requestPaint()
                        onWidthChanged: requestPaint()
                        onHeightChanged: requestPaint()
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            var r = 8
                            ctx.beginPath()
                            ctx.moveTo(r, 0)
                            ctx.lineTo(width - r, 0)
                            ctx.quadraticCurveTo(width, 0, width, r)
                            ctx.lineTo(width, height - r)
                            ctx.quadraticCurveTo(width, height, width - r, height)
                            ctx.lineTo(r, height)
                            ctx.quadraticCurveTo(0, height, 0, height - r)
                            ctx.lineTo(0, r)
                            ctx.quadraticCurveTo(0, 0, r, 0)
                            ctx.closePath()
                            var grad = ctx.createLinearGradient(0, height, width, 0)
                            grad.addColorStop(0.0, Qt.rgba(gradStart.r, gradStart.g, gradStart.b, 1.0))
                            grad.addColorStop(1.0, Qt.rgba(gradEnd.r, gradEnd.g, gradEnd.b, 1.0))
                            ctx.fillStyle = grad
                            ctx.fill()
                        }
                    }
                    Material.foreground: root.colText
                    scale: importBtn.hovered && enabled ? 1.03 : 1.0
                    Behavior on scale { NumberAnimation { duration: 100; easing.type: Easing.OutCubic } }
                    onClicked: importDialog.open()
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: root.centerPaneMinWidth
                spacing: 10

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: root.panelRadius
                    color: root.colGlassPanel
                    border.width: 1
                    border.color: root.colGlassStroke
                    clip: true

                ColumnLayout {
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    anchors.leftMargin: 18
                    anchors.rightMargin: 18
                    anchors.topMargin: 10
                    anchors.bottomMargin: 0
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 40
                        Label { text: qsTr("Browser"); color: root.colTextMuted; font.pixelSize: 13; font.weight: 600 }
                        Item { Layout.fillWidth: true }
                        Item {
                            id: viewModeSwitch
                            Layout.preferredWidth: 132
                            Layout.preferredHeight: 36

                            Rectangle {
                                id: viewModeTrack
                                anchors.fill: parent
                                radius: height / 2
                                color: Qt.rgba(root.colBgBase.r, root.colBgBase.g, root.colBgBase.b, 0.98)
                                border.width: 1
                                border.color: root.colDivider
                            }

                            Rectangle {
                                id: viewModeThumb
                                width: parent.width / 2 - 4
                                height: parent.height - 4
                                y: 2
                                x: root.gridMode ? 2 : parent.width - width - 2
                                radius: height / 2
                                color: root.colAccentPrimary
                                border.width: 1
                                border.color: Qt.rgba(
                                    root.colAccentSecondary.r,
                                    root.colAccentSecondary.g,
                                    root.colAccentSecondary.b,
                                    0.52)
                                Behavior on x { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }
                            }

                            Row {
                                anchors.fill: parent
                                spacing: 0

                                Item {
                                    width: parent.width / 2
                                    height: parent.height

                                    Image {
                                        id: gridModeIconSource
                                        anchors.centerIn: parent
                                        width: 20
                                        height: 20
                                        source: "qrc:/panel_icons/layout-grid.svg"
                                        visible: false
                                        asynchronous: true
                                    }

                                    MultiEffect {
                                        anchors.fill: gridModeIconSource
                                        source: gridModeIconSource
                                        colorization: 1.0
                                        colorizationColor: root.gridMode ? root.colBgCanvas : root.colTextMuted
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: root.gridMode = true
                                    }
                                }

                                Item {
                                    width: parent.width / 2
                                    height: parent.height

                                    Image {
                                        id: listModeIconSource
                                        anchors.centerIn: parent
                                        width: 20
                                        height: 20
                                        source: "qrc:/panel_icons/list.svg"
                                        visible: false
                                        asynchronous: true
                                    }

                                    MultiEffect {
                                        anchors.fill: listModeIconSource
                                        source: listModeIconSource
                                        colorization: 1.0
                                        colorizationColor: root.gridMode ? root.colTextMuted : root.colBgCanvas
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: root.gridMode = false
                                    }
                                }
                            }
                        }
                    }

                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Loader {
                            anchors.fill: parent
                            active: albumBackend.shownCount > 0
                            sourceComponent: gridMode ? gridComp : listComp
                        }

                        Column {
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            visible: albumBackend.shownCount === 0
                            spacing: 8
                            Label {
                                text: albumBackend.serviceReady ? qsTr("No Photos Yet") : qsTr("Open or Create a Project")
                                font.family: root.headlineFontFamily
                                color: root.colText
                                font.pixelSize: 22
                                font.weight: 700
                            }
                            Label {
                                text: albumBackend.serviceReady
                                      ? qsTr("Import your images for RAW adjustments.")
                                      : qsTr("Use File > Load Project or File > Create Project to choose .alcd files.")
                                color: root.colTextMuted
                                font.pixelSize: 12
                            }
                            Button {
                                id: emptyStateLoadButton
                                visible: !albumBackend.serviceReady
                                text: qsTr("Load Project")
                                Material.background: root.colButtonPrimary
                                Material.foreground: root.colText
                                onClicked: albumBackend.PromptAndLoadProject()
                            }
                        }
                    }
                }
                } // close album card Rectangle

            } // close center block wrapper

            // ── inspector panel + overlay resize handle ──
            Item {
                id: inspectorContainer
                Layout.fillHeight: true
                Layout.minimumWidth: 0
                Layout.maximumWidth: root.inspectorAdaptiveMaxWidth
                Layout.preferredWidth: inspectorVisible
                                       ? Math.min(root.inspectorWidth, root.inspectorAdaptiveMaxWidth)
                                       : 0
                Behavior on Layout.preferredWidth { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: root.panelRadius
                        color: root.colBgPanel
                        border.width: 0
                        clip: true

                        InspectorPanel {
                            anchors.fill: parent
                            anchors.margins: 10
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 52
                        spacing: 10

                        Button {
                            id: addSelectedBtn
                            Layout.fillWidth: true
                            Layout.preferredHeight: 52
                            text: qsTr("Add to Queue") + " (" + root.selectedCount + ")"
                            enabled: root.backendInteractive && root.selectedCount > 0
                            icon.source: "qrc:/panel_icons/queue-add.svg"
                            icon.width: 16
                            icon.height: 16
                            icon.color: root.colText
                            display: AbstractButton.TextBesideIcon
                            background: Rectangle {
                                radius: root.controlRadius
                                color: root.secondaryButtonFill(
                                    addSelectedBtn.enabled,
                                    addSelectedBtn.hovered,
                                    addSelectedBtn.down)
                                border.width: 1
                                border.color: root.colButtonSecondaryBorder
                            }
                            Material.foreground: root.colText
                            scale: addSelectedBtn.hovered && enabled ? 1.03 : 1.0
                            Behavior on scale { NumberAnimation { duration: 100; easing.type: Easing.OutCubic } }
                            onClicked: selectionState.addSelectedToExportQueue()
                        }

                        Button {
                            id: exportQueueBtn
                            Layout.fillWidth: true
                            Layout.preferredHeight: 52
                            text: qsTr("Export") + " (" + root.exportQueueCount + ")"
                            enabled: root.backendInteractive && (albumBackend.shownCount > 0 || root.exportQueueCount > 0)
                            icon.source: "qrc:/panel_icons/export.svg"
                            icon.width: 16
                            icon.height: 16
                            icon.color: root.colText
                            display: AbstractButton.TextBesideIcon
                            background: Canvas {
                                opacity: exportQueueBtn.enabled ? 1.0 : 0.5
                                property color gradStart: root.colAccentPrimary
                                property color gradEnd: root.colAccentSecondary
                                onGradStartChanged: requestPaint()
                                onGradEndChanged: requestPaint()
                                onWidthChanged: requestPaint()
                                onHeightChanged: requestPaint()
                                onPaint: {
                                    var ctx = getContext("2d")
                                    ctx.clearRect(0, 0, width, height)
                                    var r = 8
                                    ctx.beginPath()
                                    ctx.moveTo(r, 0)
                                    ctx.lineTo(width - r, 0)
                                    ctx.quadraticCurveTo(width, 0, width, r)
                                    ctx.lineTo(width, height - r)
                                    ctx.quadraticCurveTo(width, height, width - r, height)
                                    ctx.lineTo(r, height)
                                    ctx.quadraticCurveTo(0, height, 0, height - r)
                                    ctx.lineTo(0, r)
                                    ctx.quadraticCurveTo(0, 0, r, 0)
                                    ctx.closePath()
                                    var grad = ctx.createLinearGradient(0, height, width, 0)
                                    grad.addColorStop(0.0, Qt.rgba(gradStart.r, gradStart.g, gradStart.b, 1.0))
                                    grad.addColorStop(1.0, Qt.rgba(gradEnd.r, gradEnd.g, gradEnd.b, 1.0))
                                    ctx.fillStyle = grad
                                    ctx.fill()
                                }
                            }
                            Material.foreground: root.colText
                            scale: exportQueueBtn.hovered && enabled ? 1.03 : 1.0
                            Behavior on scale { NumberAnimation { duration: 100; easing.type: Easing.OutCubic } }
                            onClicked: {
                                selectionState.refreshExportPreview()
                                exportDialog.open()
                            }
                        }
                    }
                }

                Rectangle {
                    id: inspectorResizeHandle
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    width: inspectorVisible && root.inspectorAdaptiveMaxWidth > 0 ? 5 : 0
                    x: -Math.round(width / 2)
                    color: dragArea.containsMouse || dragArea.drag.active ? root.colAccentPrimary : "transparent"
                    visible: width > 0
                    z: 10
                    Behavior on width { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }
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
                            var cappedMax = Math.min(root.inspectorMaxWidth, root.inspectorAdaptiveMaxWidth)
                            var target = startWidth + delta
                            if (cappedMax >= root.inspectorMinWidth) {
                                root.inspectorWidth = Math.max(root.inspectorMinWidth, Math.min(cappedMax, target))
                            } else {
                                root.inspectorWidth = Math.max(0, Math.min(cappedMax, target))
                            }
                        }
                    }
                }
            }
        }

    }

    }

    WelcomeDialog {
        id: welcomeDialog
        z: 30
        blurSource: mainContent
        recentProjects: albumBackend.recentProjects
        languageOptions: root.languageOptions
        currentLanguageIndex: root.languageIndexForCode(languageManager.currentLanguageCode)
        serviceMessage: albumBackend.serviceMessage
        headlineFontFamily: root.headlineFontFamily
        primaryAccent: root.colButtonPrimary
        secondaryAccent: root.colAccentSecondary
        textColor: root.colText
        mutedTextColor: root.colTextMuted
        panelColor: root.colBgPanel
        panelBorderColor: root.withAlpha(root.colText, 0.08)
        overlayColor: root.colOverlay
        baseColor: root.colBgCanvas
        onLoadRequested: {
            root.dismissWelcomeForProjectLaunch()
            root.updateWelcomeDialogVisibility()
            if (!albumBackend.PromptAndLoadProject()) {
                root.welcomeDismissedForLaunch = false
                root.updateWelcomeDialogVisibility()
            }
        }
        onCreateRequested: {
            root.dismissWelcomeForProjectLaunch()
            root.updateWelcomeDialogVisibility()
            if (!albumBackend.PromptAndCreateProject()) {
                root.welcomeDismissedForLaunch = false
                root.updateWelcomeDialogVisibility()
            }
        }
        onExitRequested: Qt.quit()
        onLanguageRequested: function(languageCode) {
            languageManager.setLanguage(languageCode)
        }
        onRecentProjectRequested: function(projectPath) {
            root.dismissWelcomeForProjectLaunch()
            root.updateWelcomeDialogVisibility()
            if (!albumBackend.LoadProject(projectPath)) {
                root.welcomeDismissedForLaunch = false
                root.updateWelcomeDialogVisibility()
            }
        }
    }

    // ── Import progress overlay ──────────────────────────────────────────
    Item {
        anchors.fill: parent
        visible: albumBackend.importRunning && !albumBackend.nikonHeRecoveryActive
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
                    font.family: root.headlineFontFamily
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
                    id: importCancelButton
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
            onImageSelectionChanged: function(elementId, imageId, fileName, selected) {
                selectionState.setImageSelected(elementId, imageId, fileName, selected)
            }
            onReplaceSelection: function(items) {
                selectionState.replaceSelectedImages(items)
            }
            onContextMenuRequested: function(item, sceneX, sceneY) {
                root.openImageContextMenu(item, sceneX, sceneY)
            }
        }
    }

    Component {
        id: listComp
        ThumbnailListView {
            selectedImagesById: root.selectedImagesById
            exportQueueById: root.exportQueueById
            onImageSelectionChanged: function(elementId, imageId, fileName, selected) {
                selectionState.setImageSelected(elementId, imageId, fileName, selected)
            }
            onReplaceSelection: function(items) {
                selectionState.replaceSelectedImages(items)
            }
            onContextMenuRequested: function(item, sceneX, sceneY) {
                root.openImageContextMenu(item, sceneX, sceneY)
            }
        }
    }
}
