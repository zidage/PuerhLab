import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Dialog {
    id: root
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
    property bool exportTriggered: false

    Overlay.modal: Rectangle {
        color: "#1E1D22"
        opacity: 0.88
    }

    signal addSelectedToQueueRequested()
    signal clearQueueRequested()
    signal ensurePreviewRequested()
    signal startExportRequested(
        string outDir,
        string format,
        bool resizeEnabled,
        int maxSide,
        int quality,
        int bitDepth,
        int pngLevel,
        string tiffCompression)

    onOpened: {
        if (exportOutDir.text.length === 0) {
            exportOutDir.text = albumBackend.defaultExportFolder
        }
        ensurePreviewRequested()
        albumBackend.resetExportState()
        exportTriggered = false
    }
    onClosed: exportTriggered = false

    FolderDialog {
        id: exportFolderDialog
        title: "Select Export Folder"
        onAccepted: exportOutDir.text = selectedFolder.toString()
    }

    background: Rectangle {
        radius: 16
        color: "#1E1D22"
        border.color: "#4D4C51"
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
                text: "Export Images"
                font.pixelSize: 22
                font.weight: 700
                font.letterSpacing: -0.3
                color: "#E3DFDB"
            }
            Label {
                Layout.fillWidth: true
                text: "Export queued images using the current edit pipeline."
                wrapMode: Text.WordWrap
                color: "#7B7D7C"
                font.pixelSize: 12
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: "#38373C"
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredWidth: 440
                Layout.fillHeight: true
                radius: 12
                color: "#1E1D22"
                border.color: "#38373C"

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
                                text: "DESTINATION"
                                color: "#7B7D7C"
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
                                    placeholderText: "Select output directory..."
                                }
                                Button {
                                    text: "Browse..."
                                    enabled: !albumBackend.exportInFlight
                                    onClicked: exportFolderDialog.open()
                                }
                            }
                        }

                        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: "#38373C" }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: "FORMAT"
                                color: "#7B7D7C"
                                font.pixelSize: 10
                                font.weight: 700
                                font.letterSpacing: 1.2
                            }

                            GridLayout {
                                Layout.fillWidth: true
                                columns: 4
                                columnSpacing: 12
                                rowSpacing: 10

                                Label { text: "Format"; color: "#7B7D7C"; font.pixelSize: 12 }
                                ComboBox {
                                    id: exportFormat
                                    Layout.fillWidth: true
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

                                Label { text: "Bit depth"; color: "#7B7D7C"; font.pixelSize: 12 }
                                ComboBox {
                                    id: exportBitDepth
                                    Layout.fillWidth: true
                                    model: [
                                        { text: "8-bit", value: 8 },
                                        { text: "16-bit", value: 16 },
                                        { text: "32-bit", value: 32 }
                                    ]
                                    textRole: "text"
                                    valueRole: "value"
                                    currentIndex: 1
                                }

                                Label { text: "Quality"; color: "#7B7D7C"; font.pixelSize: 12 }
                                SpinBox {
                                    id: exportQuality
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 100
                                    value: 95
                                    editable: true
                                }

                                Label {
                                    text: "PNG level"
                                    color: "#7B7D7C"
                                    font.pixelSize: 12
                                    visible: exportFormat.currentValue === "PNG"
                                }
                                SpinBox {
                                    id: exportPngLevel
                                    Layout.fillWidth: true
                                    from: 0
                                    to: 9
                                    value: 5
                                    editable: true
                                    visible: exportFormat.currentValue === "PNG"
                                }

                                Label {
                                    text: "Compression"
                                    color: "#7B7D7C"
                                    font.pixelSize: 12
                                    visible: exportFormat.currentValue === "TIFF"
                                }
                                ComboBox {
                                    id: exportTiffComp
                                    Layout.fillWidth: true
                                    visible: exportFormat.currentValue === "TIFF"
                                    model: [
                                        { text: "None", value: "NONE" },
                                        { text: "LZW", value: "LZW" },
                                        { text: "ZIP", value: "ZIP" }
                                    ]
                                    textRole: "text"
                                    valueRole: "value"
                                    currentIndex: 0
                                }
                            }
                        }

                        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: "#38373C" }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: "RESIZE"
                                color: "#7B7D7C"
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
                                    model: [ "No", "Yes" ]
                                    currentIndex: 0
                                }

                                Label {
                                    text: "Max side"
                                    color: "#7B7D7C"
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
                radius: 12
                color: "#1E1D22"
                border.color: "#38373C"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 12

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "QUEUE"
                            color: "#7B7D7C"
                            font.pixelSize: 10
                            font.weight: 700
                            font.letterSpacing: 1.2
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: root.exportQueueCount + " image(s)"
                            color: "#7B7D7C"
                            font.pixelSize: 11
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Button {
                            Layout.fillWidth: true
                            text: "Add Selected (" + root.selectedCount + ")"
                            enabled: !albumBackend.exportInFlight && root.selectedCount > 0
                            onClicked: root.addSelectedToQueueRequested()
                        }
                        Button {
                            text: "Clear"
                            enabled: !albumBackend.exportInFlight && root.exportQueueCount > 0
                            onClicked: root.clearQueueRequested()
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        radius: 8
                        color: "#16151A"
                        border.color: "#38373C"

                        ListView {
                            anchors.fill: parent
                            anchors.margins: 6
                            clip: true
                            spacing: 3
                            model: root.exportPreviewRows
                            delegate: Rectangle {
                                width: ListView.view.width
                                height: 22
                                radius: 4
                                color: index % 2 === 0 ? "#16151A" : "#1E1D22"
                                border.color: "#38373C"
                                Label {
                                    anchors.fill: parent
                                    anchors.leftMargin: 6
                                    verticalAlignment: Text.AlignVCenter
                                    text: modelData.label
                                    color: "#E3DFDB"
                                    elide: Text.ElideRight
                                    font.pixelSize: 11
                                }
                            }
                        }
                        Label {
                            anchors.centerIn: parent
                            visible: root.exportQueueCount === 0
                            text: "Queue is empty"
                            color: "#4D4C51"
                            font.pixelSize: 12
                            font.italic: true
                        }
                    }

                    Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: "#38373C" }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Label {
                            text: "PROGRESS"
                            color: "#7B7D7C"
                            font.pixelSize: 10
                            font.weight: 700
                            font.letterSpacing: 1.2
                        }

                        Label {
                            Layout.fillWidth: true
                            text: albumBackend.exportStatus
                            wrapMode: Text.WordWrap
                            color: "#E3DFDB"
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
                            text: albumBackend.exportCompleted + "/" + albumBackend.exportTotal
                                  + " done  ·  " + albumBackend.exportSucceeded + " written"
                                  + "  ·  " + albumBackend.exportFailed + " failed"
                                  + "  ·  " + albumBackend.exportSkipped + " skipped"
                            wrapMode: Text.WordWrap
                            color: "#7B7D7C"
                            font.pixelSize: 11
                        }

                        Label {
                            Layout.fillWidth: true
                            visible: albumBackend.exportErrorSummary.length > 0
                            text: albumBackend.exportErrorSummary
                            wrapMode: Text.WordWrap
                            color: "#E3DFDB"
                            font.pixelSize: 11
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: "#38373C"
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Button {
                text: "Cancel"
                enabled: !albumBackend.exportInFlight
                onClicked: root.close()
            }
            Item { Layout.fillWidth: true }
            Label {
                visible: root.exportQueueCount > 0
                text: root.exportQueueCount + " image(s) queued"
                color: "#7B7D7C"
                font.pixelSize: 12
            }
            Button {
                highlighted: true
                text: albumBackend.exportInFlight ? "Exporting..." : "Export"
                enabled: !albumBackend.exportInFlight && root.exportQueueCount > 0
                onClicked: {
                    root.exportTriggered = true
                    root.startExportRequested(
                        exportOutDir.text,
                        exportFormat.currentValue,
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
