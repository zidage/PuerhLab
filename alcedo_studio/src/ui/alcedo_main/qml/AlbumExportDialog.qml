import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Effects

Dialog {
    id: root
    font.family: appTheme.uiFontFamily
    modal: true
    focus: true
    width: Math.min(parent ? parent.width - 40 : 1280, 1280)
    height: Math.min(parent ? parent.height - 40 : 780, 780)
    x: parent ? Math.round((parent.width - width) / 2) : 0
    y: parent ? Math.round((parent.height - height) / 2) : 0
    padding: 0
    closePolicy: albumBackend.exportInFlight
        ? Popup.NoAutoClose
        : Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property Item blurSource: null
    property int selectedCount: 0
    property int exportQueueCount: 0
    property var exportPreviewRows: []
    property int queuePreviewLimit: 36
    property bool hdrExportAvailable: false
    property bool exportTriggered: false

    readonly property color panelColor: appTheme.bgPanelColor
    readonly property color cardColor: appTheme.bgBaseColor
    readonly property color separatorColor: "#38373C"
    readonly property color textColor: appTheme.textColor
    readonly property color mutedTextColor: "#7B7D7C"
    readonly property color emptyTextColor: "#4D4C51"
    readonly property color accentColor: appTheme.accentColor
    readonly property string dataFontFamily: appTheme.dataFontFamily
    readonly property string headlineFontFamily: appTheme.headlineFontFamily
    readonly property color btnPrimary: "#457B9D"
    readonly property color btnSecondary: "#3A3F44"

    Overlay.modal: Item {
        anchors.fill: parent

        MultiEffect {
            anchors.fill: parent
            source: root.blurSource
            visible: root.blurSource !== null
            blurEnabled: true
            blur: 0.65
            blurMax: 64
            saturation: -0.22
            brightness: -0.06
        }

        Rectangle {
            anchors.fill: parent
            color: appTheme.bgDeepColor
            opacity: 0.70
        }

        MouseArea { anchors.fill: parent; hoverEnabled: true }
    }

    signal addSelectedToQueueRequested()
    signal clearQueueRequested()
    signal ensurePreviewRequested()
    signal startExportRequested(
        string outDir,
        string format,
        string hdrExportMode,
        bool resizeEnabled,
        int maxSide,
        int quality,
        int bitDepth,
        int pngLevel,
        string tiffCompression)

    readonly property bool ultraHdrSelected: hdrExportAvailable
                                             && hdrExportMode.currentValue === "ULTRA_HDR"
    readonly property string effectiveExportFormat: root.ultraHdrSelected
                                                    ? "JPEG"
                                                    : exportFormat.currentValue
    readonly property string effectiveOutDir: {
        if (!subfolderCheck.checked || subfolderName.text.trim().length === 0)
            return exportOutDir.text
        const base = exportOutDir.text.replace(/[/\\]+$/, "")
        return base + "/" + subfolderName.text.trim()
    }
    readonly property var visibleExportRows: {
        const source = root.exportPreviewRows ? root.exportPreviewRows : []
        const limit = Math.max(1, Number(root.queuePreviewLimit))
        const visibleCount = Math.min(source.length, limit)
        const rows = []
        for (let i = 0; i < visibleCount; ++i) {
            rows.push(source[i])
        }
        if (source.length > visibleCount) {
            rows.push({
                summaryRow: true,
                label: qsTr("...(%1 more)").arg(source.length - visibleCount)
            })
        }
        return rows
    }

    function exportStatusForRow(statusKey, summaryRow) {
        if (summaryRow === true)
            return ""
        if (!statusKey)
            return albumBackend.exportInFlight ? "running" : "queued"
        const map = albumBackend.exportItemStatuses
        const value = map ? map[statusKey] : ""
        if (!value || value.length === 0)
            return albumBackend.exportInFlight ? "queued" : "queued"
        return String(value)
    }

    function exportStatusText(status) {
        switch (status) {
        case "running":
            return qsTr("Exporting")
        case "succeeded":
            return qsTr("Done")
        case "failed":
            return qsTr("Failed")
        default:
            return qsTr("Queued")
        }
    }

    function exportStatusColor(status) {
        switch (status) {
        case "running":
            return root.accentColor
        case "succeeded":
            return "#60C07A"
        case "failed":
            return "#E56B6B"
        default:
            return root.mutedTextColor
        }
    }

    function bitDepthOptionsFor(formatValue) {
        switch (formatValue) {
        case "JPEG":
        case "WEBP":
            return [{ text: "8-bit", value: 8 }]
        case "PNG":
            return [{ text: "8-bit", value: 8 }, { text: "16-bit", value: 16 }]
        case "TIFF":
            return [
                { text: "8-bit",  value: 8  },
                { text: "16-bit", value: 16 },
                { text: "32-bit", value: 32 }
            ]
        case "EXR":
            return [{ text: "16-bit", value: 16 }, { text: "32-bit", value: 32 }]
        default:
            return [{ text: "8-bit", value: 8 }]
        }
    }

    function preferredBitDepthFor(formatValue) {
        switch (formatValue) {
        case "JPEG":
        case "WEBP":
            return 8
        default:
            return 16
        }
    }

    function ensureValidBitDepthSelection() {
        const options = root.bitDepthOptionsFor(root.effectiveExportFormat)
        const current = Number(exportBitDepth.currentValue)
        for (let i = 0; i < options.length; ++i) {
            if (Number(options[i].value) === current) return
        }
        const preferred = root.preferredBitDepthFor(root.effectiveExportFormat)
        for (let i = 0; i < options.length; ++i) {
            if (Number(options[i].value) === preferred) {
                exportBitDepth.currentIndex = i
                return
            }
        }
        exportBitDepth.currentIndex = 0
    }

    onOpened: {
        if (exportOutDir.text.length === 0)
            exportOutDir.text = albumBackend.defaultExportFolder
        if (!hdrExportAvailable)
            hdrExportMode.currentIndex = 1
        ensureValidBitDepthSelection()
        ensurePreviewRequested()
        albumBackend.ResetExportState()
        exportTriggered = false
    }
    onClosed: exportTriggered = false
    onHdrExportAvailableChanged: {
        if (!hdrExportAvailable)
            hdrExportMode.currentIndex = 1
        ensureValidBitDepthSelection()
    }
    onEffectiveExportFormatChanged: ensureValidBitDepthSelection()

    FolderDialog {
        id: exportFolderDialog
        title: qsTr("Select Export Folder")
        onAccepted: exportOutDir.text = selectedFolder.toString()
    }

    background: Rectangle {
        radius: 14
        color: root.panelColor
        border.width: 1
        border.color: Qt.rgba(1, 1, 1, 0.06)
        layer.enabled: true
    }

    contentItem: ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ─── Header ───────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.topMargin: 22
            Layout.leftMargin: 24
            Layout.rightMargin: 22
            Layout.bottomMargin: 14
            spacing: 0

            ColumnLayout {
                spacing: 3
                Label {
                    text: qsTr("Export Images")
                    font.family: root.headlineFontFamily
                    font.pixelSize: 20
                    font.weight: Font.Medium
                    font.letterSpacing: -0.3
                    color: root.textColor
                }
                Label {
                    text: qsTr("Configure settings for current batch")
                    font.pixelSize: 12
                    color: root.mutedTextColor
                }
            }
            Item { Layout.fillWidth: true }
            Button {
                text: "✕"
                flat: true
                enabled: !albumBackend.exportInFlight
                onClicked: root.close()
                implicitWidth: 28
                implicitHeight: 28
                font.pixelSize: 13
                opacity: hovered ? 1.0 : 0.55
                Material.foreground: root.textColor
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        // ─── Body ─────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 16
            spacing: 12

            // ── Left: settings ────────────────────────────────
            ScrollView {
                id: settingsScroll
                Layout.fillHeight: true
                Layout.fillWidth: true
                contentWidth: availableWidth
                clip: true

                ColumnLayout {
                    width: settingsScroll.availableWidth
                    spacing: 10

                    // ── Card: Destination ──────────────────────
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: destCol.implicitHeight + 28
                        radius: 8
                        color: root.cardColor

                        ColumnLayout {
                            id: destCol
                            y: 14; x: 16
                            width: parent.width - 32
                            spacing: 10

                            RowLayout {
                                spacing: 0
                                Label {
                                    text: qsTr("Destination")
                                    font.pixelSize: 13
                                    font.weight: Font.DemiBold
                                    color: root.textColor
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                TextField {
                                    id: exportOutDir
                                    Layout.fillWidth: true
                                    placeholderText: qsTr("Select output directory...")
                                    font.family: root.dataFontFamily
                                    font.pixelSize: 12
                                    enabled: !albumBackend.exportInFlight
                                }
                                Button {
                                    text: qsTr("Browse")
                                    enabled: !albumBackend.exportInFlight
                                    onClicked: exportFolderDialog.open()
                                    implicitWidth: 80
                                    Material.foreground: root.textColor
                                    background: Rectangle {
                                        radius: 8
                                        color: parent.enabled
                                               ? (parent.down   ? Qt.darker(root.btnSecondary, 1.14)
                                                : parent.hovered ? Qt.lighter(root.btnSecondary, 1.08)
                                                                 : root.btnSecondary)
                                               : Qt.rgba(root.btnSecondary.r, root.btnSecondary.g, root.btnSecondary.b, 0.5)
                                        border.width: 1
                                        border.color: Qt.rgba(root.textColor.r, root.textColor.g, root.textColor.b, 0.12)
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                CheckBox {
                                    id: subfolderCheck
                                    text: qsTr("Put in Subfolder")
                                    checked: false
                                    font.pixelSize: 12
                                    enabled: !albumBackend.exportInFlight
                                }
                                TextField {
                                    id: subfolderName
                                    visible: subfolderCheck.checked
                                    text: "Processed"
                                    font.pixelSize: 12
                                    implicitWidth: 130
                                    enabled: !albumBackend.exportInFlight
                                }
                            }
                        }
                    }

                    // ── Cards: File Settings + Quality (side by side) ──
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10

                        // Card: File Settings
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.max(fileSettingsCol.implicitHeight,
                                                            qualityCol.implicitHeight) + 28
                            radius: 8
                            color: root.cardColor

                            ColumnLayout {
                                id: fileSettingsCol
                                y: 14; x: 16
                                width: parent.width - 32
                                spacing: 10

                                Label {
                                    text: qsTr("File Settings")
                                    font.pixelSize: 13
                                    font.weight: Font.DemiBold
                                    color: root.textColor
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 5
                                    Label {
                                        text: qsTr("Image Format")
                                        color: root.mutedTextColor
                                        font.pixelSize: 11
                                    }
                                    ComboBox {
                                        id: exportFormat
                                        Layout.fillWidth: true
                                        enabled: !root.ultraHdrSelected && !albumBackend.exportInFlight
                                        model: [
                                            { text: "JPEG", value: "JPEG" },
                                            { text: "PNG",  value: "PNG"  },
                                            { text: "TIFF", value: "TIFF" },
                                            { text: "WEBP", value: "WEBP" },
                                            { text: "EXR",  value: "EXR"  }
                                        ]
                                        textRole: "text"
                                        valueRole: "value"
                                    }
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 5
                                    Label {
                                        text: qsTr("Bit Depth")
                                        color: root.mutedTextColor
                                        font.pixelSize: 11
                                    }
                                    ComboBox {
                                        id: exportBitDepth
                                        Layout.fillWidth: true
                                        enabled: root.bitDepthOptionsFor(root.effectiveExportFormat).length > 1
                                                 && !albumBackend.exportInFlight
                                        model: root.bitDepthOptionsFor(root.effectiveExportFormat)
                                        textRole: "text"
                                        valueRole: "value"
                                    }
                                }
                            }
                        }

                        // Card: Quality & Resize
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: Math.max(fileSettingsCol.implicitHeight,
                                                            qualityCol.implicitHeight) + 28
                            radius: 8
                            color: root.cardColor

                            ColumnLayout {
                                id: qualityCol
                                y: 14; x: 16
                                width: parent.width - 32
                                spacing: 10

                                RowLayout {
                                    Layout.fillWidth: true
                                    Label {
                                        text: qsTr("Quality")
                                        font.pixelSize: 13
                                        font.weight: Font.DemiBold
                                        color: root.textColor
                                    }
                                    Item { Layout.fillWidth: true }
                                    Label {
                                        visible: root.effectiveExportFormat === "JPEG"
                                                 || root.effectiveExportFormat === "WEBP"
                                        text: Math.round(exportQualitySlider.value) + "%"
                                        font.family: root.dataFontFamily
                                        font.pixelSize: 13
                                        font.weight: Font.DemiBold
                                        color: root.accentColor
                                    }
                                }

                                Slider {
                                    id: exportQualitySlider
                                    Layout.fillWidth: true
                                    visible: root.effectiveExportFormat === "JPEG"
                                             || root.effectiveExportFormat === "WEBP"
                                    from: 1; to: 100; value: 90; stepSize: 1
                                    enabled: !albumBackend.exportInFlight
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    visible: root.effectiveExportFormat === "PNG"
                                    spacing: 8
                                    Label {
                                        text: qsTr("PNG level")
                                        color: root.mutedTextColor
                                        font.pixelSize: 11
                                    }
                                    Item { Layout.fillWidth: true }
                                    SpinBox {
                                        id: exportPngLevel
                                        from: 0; to: 9; value: 5; editable: true
                                        implicitWidth: 90
                                        enabled: !albumBackend.exportInFlight
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    visible: root.effectiveExportFormat === "TIFF"
                                    spacing: 8
                                    Label {
                                        text: qsTr("Compression")
                                        color: root.mutedTextColor
                                        font.pixelSize: 11
                                    }
                                    Item { Layout.fillWidth: true }
                                    ComboBox {
                                        id: exportTiffComp
                                        model: [
                                            { text: qsTr("None"), value: "NONE" },
                                            { text: "LZW", value: "LZW" },
                                            { text: "ZIP", value: "ZIP" }
                                        ]
                                        textRole: "text"
                                        valueRole: "value"
                                        currentIndex: 0
                                        implicitWidth: 110
                                        enabled: !albumBackend.exportInFlight
                                    }
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 5
                                    Label {
                                        text: qsTr("Limit Longest Edge (px)")
                                        color: root.mutedTextColor
                                        font.pixelSize: 11
                                    }
                                    TextField {
                                        id: exportMaxSideField
                                        Layout.fillWidth: true
                                        placeholderText: "e.g. 2048"
                                        validator: IntValidator { bottom: 256; top: 16384 }
                                        inputMethodHints: Qt.ImhDigitsOnly
                                        font.family: root.dataFontFamily
                                        font.pixelSize: 12
                                        enabled: !albumBackend.exportInFlight
                                    }
                                }
                            }
                        }
                    }

                    // ── Card: Metadata & Output ────────────────
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: metadataCol.implicitHeight + 28
                        radius: 8
                        color: root.cardColor

                        ColumnLayout {
                            id: metadataCol
                            y: 14; x: 16
                            width: parent.width - 32
                            spacing: 10

                            Label {
                                text: qsTr("Metadata & Output")
                                font.pixelSize: 13
                                font.weight: Font.DemiBold
                                color: root.textColor
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label {
                                    text: qsTr("HDR Output Format")
                                    color: root.mutedTextColor
                                    font.pixelSize: 12
                                    Layout.alignment: Qt.AlignVCenter
                                }
                                Item { Layout.fillWidth: true }
                                ComboBox {
                                    id: hdrExportMode
                                    enabled: root.hdrExportAvailable && !albumBackend.exportInFlight
                                    model: [
                                        { text: qsTr("Ultra HDR"),        value: "ULTRA_HDR"            },
                                        { text: qsTr("Embed ICC Profile"), value: "EMBEDDED_PROFILE_ONLY" }
                                    ]
                                    textRole: "text"
                                    valueRole: "value"
                                    implicitWidth: 170
                                    onCurrentValueChanged: {
                                        if (currentValue === "ULTRA_HDR")
                                            exportFormat.currentIndex = 0
                                    }
                                }
                            }

                            CheckBox {
                                id: embedColorProfileCheck
                                text: qsTr("Embed Color Profile")
                                checked: true
                                enabled: !root.ultraHdrSelected && !albumBackend.exportInFlight
                                font.pixelSize: 12
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                color: root.mutedTextColor
                                font.pixelSize: 11
                                text: !root.hdrExportAvailable
                                    ? qsTr("Ultra HDR is only available when every queued item uses an HDR output EOTF (PQ or HLG). For SDR output, only ICC profile embedding is available.")
                                    : root.ultraHdrSelected
                                    ? qsTr("Ultra HDR exports are written as JPEG and include SDR fallback for legacy viewers.")
                                    : qsTr("Embed the active output ICC profile without Ultra HDR encoding. This mode keeps all export formats available.")
                            }
                        }
                    }
                }
            }

            // ── Right: Export Queue ───────────────────────────
            Rectangle {
                Layout.preferredWidth: 320
                Layout.fillHeight: true
                radius: 8
                color: root.cardColor

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: qsTr("Export Queue")
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                            color: root.textColor
                        }
                        Item { Layout.fillWidth: true }
                        Rectangle {
                            visible: root.exportQueueCount > 0
                            radius: 4
                            color: "#2C2C32"
                            implicitWidth: queueCountLbl.implicitWidth + 14
                            implicitHeight: 22
                            Label {
                                id: queueCountLbl
                                anchors.centerIn: parent
                                text: root.exportQueueCount + " " + qsTr("Items")
                                font.family: root.dataFontFamily
                                font.pixelSize: 11
                                color: root.mutedTextColor
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Button {
                            Layout.fillWidth: true
                            text: qsTr("Add Selected (%1)").arg(root.selectedCount)
                            enabled: !albumBackend.exportInFlight && root.selectedCount > 0
                            onClicked: root.addSelectedToQueueRequested()
                            Material.foreground: root.textColor
                            background: Rectangle {
                                radius: 8
                                color: parent.enabled
                                       ? (parent.down   ? Qt.darker(root.btnSecondary, 1.14)
                                        : parent.hovered ? Qt.lighter(root.btnSecondary, 1.08)
                                                         : root.btnSecondary)
                                       : Qt.rgba(root.btnSecondary.r, root.btnSecondary.g, root.btnSecondary.b, 0.5)
                                border.width: 1
                                border.color: Qt.rgba(root.textColor.r, root.textColor.g, root.textColor.b, 0.12)
                            }
                        }
                        Button {
                            text: qsTr("Clear")
                            enabled: !albumBackend.exportInFlight && root.exportQueueCount > 0
                            onClicked: root.clearQueueRequested()
                            implicitWidth: 70
                            Material.foreground: root.textColor
                            background: Rectangle {
                                radius: 8
                                color: parent.enabled
                                       ? (parent.down   ? Qt.darker(root.btnSecondary, 1.14)
                                        : parent.hovered ? Qt.lighter(root.btnSecondary, 1.08)
                                                         : root.btnSecondary)
                                       : Qt.rgba(root.btnSecondary.r, root.btnSecondary.g, root.btnSecondary.b, 0.5)
                                border.width: 1
                                border.color: Qt.rgba(root.textColor.r, root.textColor.g, root.textColor.b, 0.12)
                            }
                        }
                    }

                    Item {
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        ListView {
                            anchors.fill: parent
                            clip: true
                            spacing: 2
                            model: root.visibleExportRows

                            delegate: RowLayout {
                                width: ListView.view.width
                                height: 50
                                spacing: 10
                                readonly property bool summaryRow: modelData.summaryRow === true
                                readonly property string itemStatus: root.exportStatusForRow(
                                    modelData.statusKey, summaryRow)

                                Rectangle {
                                    width: 34; height: 34
                                    radius: 4
                                    color: "#28282E"
                                    Image {
                                        anchors.centerIn: parent
                                        width: 20
                                        height: 20
                                        sourceSize.width: 20
                                        sourceSize.height: 20
                                        fillMode: Image.PreserveAspectFit
                                        source: "qrc:/panel_icons/image.svg"
                                        opacity: 0.9
                                    }
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 2
                                    Label {
                                        Layout.fillWidth: true
                                        text: modelData.label
                                        elide: Text.ElideRight
                                        color: root.textColor
                                        font.family: root.dataFontFamily
                                        font.pixelSize: 12
                                    }
                                    RowLayout {
                                        spacing: 5
                                        Rectangle {
                                            visible: !summaryRow
                                            width: 7; height: 7; radius: 3.5
                                            color: root.exportStatusColor(itemStatus)
                                        }
                                        Label {
                                            visible: !summaryRow
                                            text: root.exportStatusText(itemStatus)
                                            color: root.exportStatusColor(itemStatus)
                                            font.pixelSize: 11
                                        }
                                    }
                                }
                            }
                        }

                        Label {
                            anchors.centerIn: parent
                            visible: root.exportQueueCount === 0
                            text: qsTr("Queue is empty")
                            color: root.emptyTextColor
                            font.pixelSize: 12
                            font.italic: true
                        }
                    }
                }
            }
        }

        // ─── Footer separator ─────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        // ─── Footer ───────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.topMargin: 14
            Layout.bottomMargin: 14
            Layout.leftMargin: 24
            Layout.rightMargin: 22
            spacing: 16

            ColumnLayout {
                spacing: 5
                RowLayout {
                    spacing: 8
                    Label {
                        text: qsTr("Export Progress")
                        color: root.mutedTextColor
                        font.pixelSize: 12
                    }
                    Label {
                        text: albumBackend.exportCompleted + "/"
                              + (albumBackend.exportTotal > 0
                                 ? albumBackend.exportTotal
                                 : root.exportQueueCount)
                        color: root.mutedTextColor
                        font.family: root.dataFontFamily
                        font.pixelSize: 12
                    }
                }
                ProgressBar {
                    implicitWidth: 240
                    value: albumBackend.exportTotal > 0
                           ? albumBackend.exportCompleted / albumBackend.exportTotal
                           : 0
                }
            }

            Label {
                visible: albumBackend.exportErrorSummary.length > 0
                text: albumBackend.exportErrorSummary
                wrapMode: Text.WordWrap
                color: root.textColor
                font.family: root.dataFontFamily
                font.pixelSize: 11
                Layout.fillWidth: true
            }

            Item { Layout.fillWidth: true }

            Button {
                text: (root.exportTriggered && !albumBackend.exportInFlight)
                      ? qsTr("Close")
                      : qsTr("Cancel")
                enabled: !albumBackend.exportInFlight
                onClicked: root.close()
                Material.foreground: root.textColor
                background: Rectangle {
                    radius: 8
                    color: parent.enabled
                           ? (parent.down   ? Qt.darker(root.btnSecondary, 1.14)
                            : parent.hovered ? Qt.lighter(root.btnSecondary, 1.08)
                                             : root.btnSecondary)
                           : Qt.rgba(root.btnSecondary.r, root.btnSecondary.g, root.btnSecondary.b, 0.5)
                    border.width: 1
                    border.color: Qt.rgba(root.textColor.r, root.textColor.g, root.textColor.b, 0.12)
                }
            }
            Button {
                highlighted: true
                text: albumBackend.exportInFlight
                      ? qsTr("Exporting...")
                      : root.exportQueueCount === 1
                        ? qsTr("Export 1 File")
                        : qsTr("Export %1 Files").arg(root.exportQueueCount)
                enabled: !albumBackend.exportInFlight && root.exportQueueCount > 0
                Material.foreground: root.textColor
                background: Rectangle {
                    radius: 8
                    color: parent.enabled
                           ? (parent.down   ? Qt.darker(root.btnPrimary, 1.18)
                            : parent.hovered ? Qt.lighter(root.btnPrimary, 1.08)
                                             : root.btnPrimary)
                           : Qt.rgba(root.btnPrimary.r, root.btnPrimary.g, root.btnPrimary.b, 0.45)
                }
                onClicked: {
                    root.exportTriggered = true
                    const hasResize = exportMaxSideField.text.trim().length > 0
                    const maxSide   = hasResize ? parseInt(exportMaxSideField.text) : 0
                    root.startExportRequested(
                        root.effectiveOutDir,
                        exportFormat.currentValue,
                        hdrExportMode.currentValue,
                        hasResize,
                        maxSide,
                        Math.round(exportQualitySlider.value),
                        Number(exportBitDepth.currentValue),
                        exportPngLevel.value,
                        exportTiffComp.currentValue)
                }
            }
        }
    }
}
