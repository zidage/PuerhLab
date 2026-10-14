import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Import progress overlay, laid out like the agent session card: a state lamp
// and a state line beside the title, a gradient that turns around the outline
// while the import works, and a close (X) button at the top right that cancels
// the import.
//
// The ring and the count inside it show photos imported out of the photos
// expected. Files rejected as not RAW (or failed) leave the expected total, so
// the total shrinks while the import runs instead of stalling the ring. The
// line below the ring counts files checked out of all files handed to the
// import. "preparing" registers the files, "reading" reads their metadata, and
// "finalizing" writes the result to the project; the ring spins while the
// amount of remaining work is unknown (start of preparing, finalizing).
//
// Cancel takes effect at once: the lamp, ring and outline turn gray, the state
// line reads "Canceling", and the X is disabled. Files already being read
// finish and the photos imported so far are saved; queued files are skipped.
Item {
    id: root
    objectName: "importProgressOverlay"
    property var theme: null
    property Item blurSource: null
    anchors.fill: parent

    readonly property var handler: appModules.importExport
    readonly property string phase: handler.importPhase
    readonly property bool canceling: handler.importCanceling
    readonly property int total: handler.importTotal
    readonly property int completed: handler.importCompleted
    readonly property int failed: handler.importFailed
    readonly property int canceled: Math.min(handler.importCanceled, failed)
    readonly property int unsupported: Math.min(handler.importUnsupported, failed - canceled)
    readonly property int excluded: Math.min(handler.importExcluded,
                                             failed - canceled - unsupported)
    readonly property int errors: failed - canceled - unsupported - excluded
    readonly property int expected: Math.max(0, total - failed)
    readonly property int checked: Math.min(total, completed + failed - canceled)
    readonly property bool finalizing: phase === "finalizing"
    readonly property bool progressUnknown: finalizing
                                            || (phase === "preparing" && checked === 0)
    readonly property bool canCancel: !canceling && !finalizing

    readonly property color textColor: root.theme ? root.theme.colText : appTheme.textColor
    readonly property color mutedColor: root.theme ? root.theme.colTextMuted
                                                   : appTheme.textMutedColor
    readonly property color stateColor: canceling ? mutedColor
                                                  : appTheme.backgroundTaskWorkingColor

    function formatCount(value) {
        return Number(value).toLocaleString(Qt.locale(), 'f', 0)
    }

    function stateLabel() {
        if (root.canceling)
            return qsTr("Canceling")
        if (root.finalizing)
            return qsTr("Saving")
        if (root.phase === "preparing")
            return qsTr("Preparing")
        return qsTr("Reading")
    }

    function phaseText() {
        if (root.canceling && root.finalizing)
            return qsTr("Saving the photos imported before the cancel...")
        if (root.canceling)
            return qsTr("Finishing the files already being read. Queued files are skipped.")
        if (root.finalizing)
            return qsTr("Saving to the library...")
        if (root.phase === "preparing")
            return qsTr("Preparing files... %1 of %2")
                   .arg(root.formatCount(root.handler.importPrepared))
                   .arg(root.formatCount(root.total))
        return qsTr("Reading photos... %1 of %2 files checked")
               .arg(root.formatCount(root.checked))
               .arg(root.formatCount(root.total))
    }

    BlurredOverlay {
        anchors.fill: parent
        blurSource: root.blurSource
        overlayColor: root.theme ? root.theme.colOverlay : appTheme.overlayColor

        Rectangle {
            id: importCard
            anchors.centerIn: parent
            width: Math.min(parent.width - appTheme.spaceXl * 2, 420)
            height: importDialogContent.implicitHeight + appTheme.spaceLg * 2
            radius: appTheme.panelRadius
            color: root.theme ? root.theme.colBgDeep : appTheme.bgDeepColor
            border.width: 0

            RotatingTaskBorder {
                objectName: "importProgressOutline"
                anchors.fill: parent
                radius: importCard.radius
                color: root.stateColor
                running: root.visible && !root.canceling
            }

            ColumnLayout {
                id: importDialogContent
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: appTheme.spaceLg
                spacing: appTheme.spaceMd

                RowLayout {
                    Layout.fillWidth: true
                    spacing: appTheme.spaceSm

                    Rectangle {
                        Layout.preferredWidth: appTheme.spaceSm
                        Layout.preferredHeight: appTheme.spaceSm
                        Layout.alignment: Qt.AlignVCenter
                        radius: width / 2
                        color: root.stateColor
                        Accessible.ignored: true
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 0

                        Label {
                            Layout.fillWidth: true
                            text: qsTr("Importing Photos")
                            elide: Text.ElideRight
                            font.family: root.theme ? root.theme.headlineFontFamily
                                                    : appTheme.headlineFontFamily
                            font.pixelSize: appTheme.fontSizeTitle
                            font.weight: appTheme.fontWeightStrong
                            color: root.textColor
                        }

                        Label {
                            objectName: "importProgressState"
                            Layout.fillWidth: true
                            text: qsTr("Import · %1").arg(root.stateLabel())
                            elide: Text.ElideRight
                            color: root.mutedColor
                            font.family: appTheme.uiFontFamily
                            font.pixelSize: appTheme.fontSizeCaption
                            font.weight: appTheme.fontWeightStrong
                        }
                    }

                    IconActionButton {
                        objectName: "importCancelButton"
                        Layout.alignment: Qt.AlignVCenter
                        compact: true
                        enabled: root.canCancel
                        iconSrc: "qrc:/panel_icons/close.svg"
                        fillIdle: root.theme ? root.theme.colBgDeep : appTheme.bgDeepColor
                        actionName: qsTr("Cancel import")
                        onClicked: root.handler.CancelImport()
                    }
                }

                Item {
                    Layout.alignment: Qt.AlignHCenter
                    Layout.preferredWidth: 160
                    Layout.preferredHeight: 160

                    ImportProgressRing {
                        objectName: "importProgressRing"
                        anchors.fill: parent
                        ringWidth: 12
                        trackColor: root.theme ? root.theme.colHover : appTheme.hoverColor
                        fillColor: root.canceling
                                   ? root.mutedColor
                                   : (root.theme ? root.theme.colAccentPrimary
                                                 : appTheme.accentColor)
                        indeterminate: root.progressUnknown
                        running: root.visible
                        progress: root.expected > 0 ? root.completed / root.expected : 0
                    }

                    ColumnLayout {
                        anchors.centerIn: parent
                        width: parent.width - 2 * (12 + appTheme.spaceMd)
                        spacing: 0

                        Label {
                            objectName: "importProgressCount"
                            Layout.fillWidth: true
                            horizontalAlignment: Text.AlignHCenter
                            text: root.formatCount(root.completed)
                            fontSizeMode: Text.HorizontalFit
                            minimumPixelSize: appTheme.fontSizeTitle
                            font.family: root.theme ? root.theme.dataFontFamily
                                                    : appTheme.dataFontFamily
                            font.pixelSize: appTheme.fontSizeHeadline + 6
                            font.weight: appTheme.fontWeightStrong
                            color: root.textColor
                        }

                        Label {
                            Layout.fillWidth: true
                            horizontalAlignment: Text.AlignHCenter
                            wrapMode: Text.Wrap
                            text: qsTr("of %1 photos imported").arg(root.formatCount(root.expected))
                            color: root.mutedColor
                            font.pixelSize: appTheme.fontSizeCaption
                        }
                    }
                }

                Label {
                    objectName: "importProgressPhase"
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                    text: root.phaseText()
                    color: root.mutedColor
                    font.family: appTheme.uiFontFamily
                    font.pixelSize: appTheme.fontSizeCaption
                }

                Label {
                    Layout.fillWidth: true
                    visible: root.unsupported > 0
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("%1 file(s) skipped because they are not supported images")
                          .arg(root.formatCount(root.unsupported))
                    color: root.mutedColor
                    font.pixelSize: appTheme.fontSizeCaption
                }

                Label {
                    objectName: "importProgressExcluded"
                    Layout.fillWidth: true
                    visible: root.excluded > 0
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("%1 file(s) skipped because their type is not selected")
                          .arg(root.formatCount(root.excluded))
                    color: root.mutedColor
                    font.pixelSize: appTheme.fontSizeCaption
                }

                Label {
                    objectName: "importProgressCanceled"
                    Layout.fillWidth: true
                    visible: root.canceled > 0
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("%1 file(s) not read because the import was canceled")
                          .arg(root.formatCount(root.canceled))
                    color: root.mutedColor
                    font.pixelSize: appTheme.fontSizeCaption
                }

                Label {
                    Layout.fillWidth: true
                    visible: root.errors > 0
                    wrapMode: Text.Wrap
                    horizontalAlignment: Text.AlignHCenter
                    text: qsTr("%1 file(s) failed").arg(root.formatCount(root.errors))
                    color: root.theme ? root.theme.colDanger : appTheme.dangerColor
                    font.family: root.theme ? root.theme.dataFontFamily : appTheme.dataFontFamily
                    font.pixelSize: appTheme.fontSizeCaption
                }
            }
        }
    }
}
