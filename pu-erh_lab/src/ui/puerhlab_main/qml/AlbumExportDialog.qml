import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Dialog {
    id: root
    font.family: appTheme.uiFontFamily
    modal: true
    focus: true
    width: Math.min(parent ? parent.width - 36 : 820, 820)
    height: Math.min(parent ? parent.height - 40 : 640, 640)
    x: parent ? Math.round((parent.width - width) / 2) : 0
    y: parent ? Math.round((parent.height - height) / 2) : 0
    padding: 0
    closePolicy: albumBackend.exportInFlight
        ? Popup.NoAutoClose
        : Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property int selectedCount: 0
    property int exportQueueCount: 0
    property var exportPreviewRows: []
    property bool hdrExportAvailable: false
    property bool exportTriggered: false
    readonly property color overlayColor: appTheme.bgDeepColor
    readonly property color panelColor: appTheme.bgPanelColor
    readonly property color sectionColor: appTheme.bgPanelColor
    readonly property color separatorColor: "#38373C"
    readonly property color textColor: appTheme.textColor
    readonly property color mutedTextColor: "#7B7D7C"
    readonly property color emptyTextColor: "#4D4C51"
    readonly property string dataFontFamily: appTheme.dataFontFamily

    Overlay.modal: Rectangle {
        color: root.overlayColor
        opacity: 0.88
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

    function bitDepthOptionsFor(formatValue) {
        switch (formatValue) {
        case "JPEG":
        case "WEBP":
            return [
                { text: "8-bit", value: 8 }
            ]
        case "PNG":
            return [
                { text: "8-bit", value: 8 },
                { text: "16-bit", value: 16 }
            ]
        case "TIFF":
            return [
                { text: "8-bit", value: 8 },
                { text: "16-bit", value: 16 },
                { text: "32-bit", value: 32 }
            ]
        case "EXR":
            return [
                { text: "16-bit", value: 16 },
                { text: "32-bit", value: 32 }
            ]
        default:
            return [
                { text: "8-bit", value: 8 }
            ]
        }
    }

    function preferredBitDepthFor(formatValue) {
        switch (formatValue) {
        case "JPEG":
        case "WEBP":
            return 8
        case "PNG":
        case "TIFF":
        case "EXR":
            return 16
        default:
            return 8
        }
    }

    function ensureValidBitDepthSelection() {
        const options = root.bitDepthOptionsFor(root.effectiveExportFormat)
        const current = Number(exportBitDepth.currentValue)
        for (let i = 0; i < options.length; ++i) {
            if (Number(options[i].value) === current) {
                return
            }
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
        if (exportOutDir.text.length === 0) {
            exportOutDir.text = albumBackend.defaultExportFolder
        }
        if (!hdrExportAvailable) {
            hdrExportMode.currentIndex = 1
        }
        ensureValidBitDepthSelection()
        ensurePreviewRequested()
        albumBackend.ResetExportState()
        exportTriggered = false
    }
    onClosed: exportTriggered = false
    onHdrExportAvailableChanged: {
        if (!hdrExportAvailable) {
            hdrExportMode.currentIndex = 1
        }
        ensureValidBitDepthSelection()
    }
    onEffectiveExportFormatChanged: ensureValidBitDepthSelection()

    FolderDialog {
        id: exportFolderDialog
        title: qsTr("Select Export Folder")
        onAccepted: exportOutDir.text = selectedFolder.toString()
    }

    background: Rectangle {
        radius: 12
        color: root.panelColor
        border.width: 0
        layer.enabled: true
    }

    contentItem: ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4
            Label {
                Layout.fillWidth: true
                text: qsTr("Export Images")
                font.pixelSize: 22
                font.weight: 700
                font.letterSpacing: -0.3
                color: root.textColor
            }
            Label {
                Layout.fillWidth: true
                text: qsTr("Export queued images using the current edit pipeline.")
                wrapMode: Text.WordWrap
                color: root.mutedTextColor
                font.pixelSize: 12
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredWidth: 440
                Layout.fillHeight: true
                radius: 8
                color: root.sectionColor
                border.width: 0

                ScrollView {
                    id: settingsScroll
                    anchors.fill: parent
                    anchors.margins: 14
                    contentWidth: availableWidth
                    clip: true

                    ColumnLayout {
                        width: settingsScroll.availableWidth
                        spacing: 16

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: qsTr("DESTINATION")
                                color: root.mutedTextColor
                                font.pixelSize: 10
                                font.weight: 700
                                font.letterSpacing: 1.2
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                TextField {
                                    id: exportOutDir
                                    Layout.fillWidth: true
                                    placeholderText: qsTr("Select output directory...")
                                }
                                Button {
                                    text: qsTr("Browse...")
                                    enabled: !albumBackend.exportInFlight
                                    onClicked: exportFolderDialog.open()
                                }
                            }
                        }

                        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: root.separatorColor }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: qsTr("HDR")
                                color: root.mutedTextColor
                                font.pixelSize: 10
                                font.weight: 700
                                font.letterSpacing: 1.2
                            }

                            GridLayout {
                                Layout.fillWidth: true
                                columns: 2
                                columnSpacing: 12
                                rowSpacing: 10

                                Label { text: qsTr("Mode"); color: root.mutedTextColor; font.pixelSize: 12 }
                                ComboBox {
                                    id: hdrExportMode
                                    Layout.fillWidth: true
                                    enabled: hdrExportAvailable && !albumBackend.exportInFlight
                                    model: [
                                        { text: qsTr("Ultra HDR"), value: "ULTRA_HDR" },
                                        { text: qsTr("Embed ICC Profile"), value: "EMBEDDED_PROFILE_ONLY" }
                                    ]
                                    textRole: "text"
                                    valueRole: "value"
                                    onCurrentValueChanged: {
                                        if (currentValue === "ULTRA_HDR") {
                                            exportFormat.currentIndex = 0
                                        }
                                    }
                                }
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

                        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: root.separatorColor }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: qsTr("FORMAT")
                                color: root.mutedTextColor
                                font.pixelSize: 10
                                font.weight: 700
                                font.letterSpacing: 1.2
                            }

                            GridLayout {
                                Layout.fillWidth: true
                                columns: 4
                                columnSpacing: 12
                                rowSpacing: 10

                                Label { text: qsTr("Format"); color: root.mutedTextColor; font.pixelSize: 12 }
                                ComboBox {
                                    id: exportFormat
                                    Layout.fillWidth: true
                                    enabled: !root.ultraHdrSelected
                                    model: [
                                        { text: "JPEG", value: "JPEG" },
                                        { text: "PNG", value: "PNG" },
                                        { text: "TIFF", value: "TIFF" },
                                        { text: "WEBP", value: "WEBP" },
                                        { text: "EXR", value: "EXR" }
                                    ]
                                    textRole: "text"
                                    valueRole: "value"
                                }

                                Label { text: qsTr("Bit depth"); color: root.mutedTextColor; font.pixelSize: 12 }
                                ComboBox {
                                    id: exportBitDepth
                                    Layout.fillWidth: true
                                    enabled: root.bitDepthOptionsFor(root.effectiveExportFormat).length > 1
                                             && !albumBackend.exportInFlight
                                    model: root.bitDepthOptionsFor(root.effectiveExportFormat)
                                    textRole: "text"
                                    valueRole: "value"
                                }

                                Label {
                                    text: qsTr("Quality")
                                    color: root.mutedTextColor
                                    font.pixelSize: 12
                                    visible: root.effectiveExportFormat === "JPEG"
                                             || root.effectiveExportFormat === "WEBP"
                                }
                                SpinBox {
                                    id: exportQuality
                                    Layout.fillWidth: true
                                    visible: root.effectiveExportFormat === "JPEG"
                                             || root.effectiveExportFormat === "WEBP"
                                    from: 1
                                    to: 100
                                    value: 95
                                    editable: true
                                }

                                Label {
                                    text: qsTr("PNG level")
                                    color: root.mutedTextColor
                                    font.pixelSize: 12
                                    visible: root.effectiveExportFormat === "PNG"
                                }
                                SpinBox {
                                    id: exportPngLevel
                                    Layout.fillWidth: true
                                    from: 0
                                    to: 9
                                    value: 5
                                    editable: true
                                    visible: root.effectiveExportFormat === "PNG"
                                }

                                Label {
                                    text: qsTr("Compression")
                                    color: root.mutedTextColor
                                    font.pixelSize: 12
                                    visible: root.effectiveExportFormat === "TIFF"
                                }
                                ComboBox {
                                    id: exportTiffComp
                                    Layout.fillWidth: true
                                    visible: root.effectiveExportFormat === "TIFF"
                                    model: [
                                        { text: qsTr("None"), value: "NONE" },
                                        { text: "LZW", value: "LZW" },
                                        { text: "ZIP", value: "ZIP" }
                                    ]
                                    textRole: "text"
                                    valueRole: "value"
                                    currentIndex: 0
                                }
                            }

                            Label {
                                Layout.fillWidth: true
                                visible: root.effectiveExportFormat === "JPEG"
                                         && !root.ultraHdrSelected
                                         && root.hdrExportAvailable
                                wrapMode: Text.WordWrap
                                color: root.mutedTextColor
                                font.pixelSize: 11
                                text: qsTr("PQ or HLG JPEG exports can be written as Ultra HDR with SDR fallback for legacy viewers.")
                            }
                        }

                        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: root.separatorColor }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: qsTr("RESIZE")
                                color: root.mutedTextColor
                                font.pixelSize: 10
                                font.weight: 700
                                font.letterSpacing: 1.2
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                ComboBox {
                                    id: exportResize
                                    Layout.preferredWidth: 100
                                    model: [ qsTr("No"), qsTr("Yes") ]
                                    currentIndex: 0
                                }

                                Label {
                                    text: qsTr("Max side")
                                    color: root.mutedTextColor
                                    font.pixelSize: 12
                                    visible: exportResize.currentIndex === 1
                                }
                                SpinBox {
                                    id: exportMaxSide
                                    Layout.fillWidth: true
                                    from: 256
                                    to: 16384
                                    value: 4096
                                    editable: true
                                    visible: exportResize.currentIndex === 1
                                }
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredWidth: 300
                Layout.fillHeight: true
                radius: 8
                color: root.sectionColor
                border.width: 0

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 12

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: qsTr("QUEUE")
                            color: root.mutedTextColor
                            font.pixelSize: 10
                            font.weight: 700
                            font.letterSpacing: 1.2
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: qsTr("%1 image(s)").arg(root.exportQueueCount)
                            color: root.mutedTextColor
                            font.family: root.dataFontFamily
                            font.pixelSize: 11
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
                        }
                        Button {
                            text: qsTr("Clear")
                            enabled: !albumBackend.exportInFlight && root.exportQueueCount > 0
                            onClicked: root.clearQueueRequested()
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 6
                        color: root.sectionColor
                        border.width: 0

                        ListView {
                            anchors.fill: parent
                            anchors.margins: 6
                            clip: true
                            spacing: 3
                            model: root.exportPreviewRows
                            delegate: Rectangle {
                                width: ListView.view.width
                                height: 22
                                radius: 3
                                color: root.sectionColor
                                border.width: 0
                                Label {
                                    anchors.fill: parent
                                    anchors.leftMargin: 6
                                    verticalAlignment: Text.AlignVCenter
                                    text: modelData.label
                                    color: root.textColor
                                    elide: Text.ElideRight
                                    font.family: root.dataFontFamily
                                    font.pixelSize: 11
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

                    Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: root.separatorColor }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Label {
                            text: qsTr("PROGRESS")
                            color: root.mutedTextColor
                            font.pixelSize: 10
                            font.weight: 700
                            font.letterSpacing: 1.2
                        }

                        Label {
                            Layout.fillWidth: true
                            text: albumBackend.exportStatus
                            wrapMode: Text.WordWrap
                            color: root.textColor
                            font.family: root.dataFontFamily
                            font.pixelSize: 12
                        }

                        ProgressBar {
                            Layout.fillWidth: true
                            value: albumBackend.exportTotal > 0
                                ? albumBackend.exportCompleted / albumBackend.exportTotal
                                : 0
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: albumBackend.exportTotal > 0 || albumBackend.exportSkipped > 0
                            text: qsTr("%1/%2 done  ·  %3 written  ·  %4 failed  ·  %5 skipped")
                                .arg(albumBackend.exportCompleted)
                                .arg(albumBackend.exportTotal)
                                .arg(albumBackend.exportSucceeded)
                                .arg(albumBackend.exportFailed)
                                .arg(albumBackend.exportSkipped)
                            wrapMode: Text.WordWrap
                            color: root.mutedTextColor
                            font.family: root.dataFontFamily
                            font.pixelSize: 11
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: albumBackend.exportErrorSummary.length > 0
                            text: albumBackend.exportErrorSummary
                            wrapMode: Text.WordWrap
                            color: root.textColor
                            font.family: root.dataFontFamily
                            font.pixelSize: 11
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: root.separatorColor
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Button {
                text: (root.exportTriggered && !albumBackend.exportInFlight)
                    ? qsTr("Close")
                    : qsTr("Cancel")
                enabled: !albumBackend.exportInFlight
                onClicked: root.close()
            }
            Item { Layout.fillWidth: true }
            Label {
                visible: root.exportQueueCount > 0
                text: qsTr("%1 image(s) queued").arg(root.exportQueueCount)
                color: root.mutedTextColor
                font.family: root.dataFontFamily
                font.pixelSize: 12
            }
            Button {
                highlighted: true
                text: albumBackend.exportInFlight ? qsTr("Exporting...") : qsTr("Export")
                enabled: !albumBackend.exportInFlight && root.exportQueueCount > 0
                onClicked: {
                    root.exportTriggered = true
                    root.startExportRequested(
                        exportOutDir.text,
                        exportFormat.currentValue,
                        hdrExportMode.currentValue,
                        exportResize.currentIndex === 1,
                        exportMaxSide.value,
                        exportQuality.value,
                        Number(exportBitDepth.currentValue),
                        exportPngLevel.value,
                        exportTiffComp.currentValue)
                }
            }
        }
    }
}
