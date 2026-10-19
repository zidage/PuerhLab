import QtQuick
import QtQuick.Controls

// One filmstrip tile. Two independent states:
// - editing: the editor shows this image (bone well + ink frame, the strongest state);
// - selected: the image is in the shared Library selection (light gray well).
// The file name and the star rating sit in a fixed two-line caption below the
// image (stars on the upper line, right-aligned; the file name on the lower
// line), so they never cover the photo.
Item {
    id: root
    objectName: "editorFilmstripTileCard"

    property string fileName: ""
    property int imageIndex: 0
    property string liveThumbUrl: ""
    property bool liveThumbLoading: false
    property bool thumbnailReady: liveThumbUrl.length > 0
    property bool thumbnailProblemState: false
    property string thumbnailProblemText: ""
    property bool editing: false
    property bool selected: false
    property bool hovered: false
    property bool hasFocusFrame: false
    property bool saving: false
    property bool renderBusy: false
    property int displayRating: 0
    property int thumbnailMaxEdge: 512

    readonly property string tileState: editing ? "editing" : (selected ? "selected" : "idle")

    Accessible.ignored: true

    function ratingStars(value) {
        const rating = Math.max(0, Math.min(5, Math.round(Number(value))))
        let text = ""
        for (let i = 1; i <= 5; ++i) {
            text += i <= rating ? "★" : "☆"
        }
        return text
    }

    // Borderless card shell (matches ThumbnailGridView delegate): the card has no
    // outline of its own; the thumbnailFrame is the only frame. Editing is carried
    // by the bone fill plus the frame's ink border; Library selection by the light
    // gray fill only.
    Rectangle {
        id: tile
        objectName: "editorFilmstripTileSurface"
        anchors.fill: parent
        radius: appTheme.panelRadius
        color: root.editing
               ? appTheme.editorListSelectedFillColor
               : (root.selected
                  ? appTheme.filmstripSelectedFillColor
                  : (root.hovered ? appTheme.buttonHoveredFillColor : appTheme.cardSurfaceColor))
        Behavior on color { ColorAnimation { duration: 120 } }

        Rectangle {
            id: thumbnailFrame
            objectName: "editorFilmstripThumbnailFrame"
            anchors.fill: parent
            anchors.margins: appTheme.spaceXs
            radius: appTheme.controlRadiusSmall
            color: appTheme.bgBaseColor
            border.width: 1
            border.color: root.editing
                         ? appTheme.editorListSelectedInkColor : appTheme.dividerColor
            clip: true

            Image {
                id: thumbnailImage
                objectName: "editorFilmstripThumbnailImage"
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: captionBar.top
                anchors.margins: 1
                source: root.liveThumbUrl
                sourceSize.width: root.thumbnailMaxEdge
                sourceSize.height: root.thumbnailMaxEdge
                asynchronous: true
                fillMode: Image.PreserveAspectFit
                visible: root.thumbnailReady
            }

            BusyIndicator {
                anchors.centerIn: thumbnailImage
                width: appTheme.iconOpticalSize
                height: appTheme.iconOpticalSize
                visible: root.liveThumbLoading
                running: visible
            }

            Label {
                anchors.centerIn: thumbnailImage
                width: Math.max(1, parent.width - appTheme.spaceLg)
                text: root.thumbnailProblemState
                      ? root.thumbnailProblemText : qsTr("No thumbnail")
                visible: !root.thumbnailReady && !root.liveThumbLoading
                color: root.thumbnailProblemState
                       ? appTheme.dangerColor : appTheme.textMutedColor
                font.pixelSize: appTheme.fontSizeCaption
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                maximumLineCount: 2
                elide: Text.ElideRight
            }

            Row {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.margins: appTheme.spaceSm
                spacing: appTheme.spaceXs
                visible: root.editing && (root.saving || root.renderBusy)

                Rectangle {
                    objectName: "editorFilmstripSavingBadge"
                    visible: root.saving
                    width: savingLabel.implicitWidth + appTheme.spaceSm
                    height: savingLabel.implicitHeight + appTheme.spaceXs
                    radius: appTheme.badgeRadius
                    color: appTheme.editorListSelectedInkColor
                    Label {
                        id: savingLabel
                        anchors.centerIn: parent
                        text: qsTr("Saving")
                        color: appTheme.editorListSelectedFillColor
                        font.pixelSize: appTheme.fontSizeCaption
                        font.weight: appTheme.fontWeightStrong
                    }
                }

                Rectangle {
                    objectName: "editorFilmstripRenderBadge"
                    visible: root.renderBusy
                    width: renderLabel.implicitWidth + appTheme.spaceSm
                    height: renderLabel.implicitHeight + appTheme.spaceXs
                    radius: appTheme.badgeRadius
                    color: appTheme.editorListSelectedInkColor
                    Label {
                        id: renderLabel
                        anchors.centerIn: parent
                        text: qsTr("Rendering")
                        color: appTheme.editorListSelectedFillColor
                        font.pixelSize: appTheme.fontSizeCaption
                        font.weight: appTheme.fontWeightStrong
                    }
                }
            }

            // Fixed two-line caption on the bottom edge of the frame, below the
            // image. A rated image shows its stars on the upper line,
            // right-aligned; the file name has the full width of the lower line.
            // The height does not depend on the rating, so every image in the
            // strip has the same size.
            Rectangle {
                id: captionBar
                objectName: "editorFilmstripCaptionBar"
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.margins: 1
                height: appTheme.lineHeightCaption * 2 + appTheme.spaceXs * 2
                color: appTheme.thumbnailCaptionBarColor
                z: 3

                Label {
                    id: ratingLabel
                    objectName: "editorFilmstripRatingLabel"
                    anchors.top: parent.top
                    anchors.right: parent.right
                    anchors.topMargin: appTheme.spaceXs
                    anchors.rightMargin: appTheme.spaceSm
                    height: appTheme.lineHeightCaption
                    visible: root.displayRating > 0
                    text: root.ratingStars(root.displayRating)
                    color: appTheme.textColor
                    font.family: appTheme.dataFontFamily
                    font.pixelSize: appTheme.fontSizeCaption
                    horizontalAlignment: Text.AlignRight
                    verticalAlignment: Text.AlignVCenter
                }

                Label {
                    id: fileNameLabel
                    objectName: "editorFilmstripFileNameLabel"
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    anchors.leftMargin: appTheme.spaceSm
                    anchors.rightMargin: appTheme.spaceSm
                    anchors.bottomMargin: appTheme.spaceXs
                    height: appTheme.lineHeightCaption
                    text: root.fileName.length > 0
                          ? root.fileName : qsTr("Image %1").arg(root.imageIndex + 1)
                    color: appTheme.textColor
                    font.pixelSize: appTheme.fontSizeCaption
                    font.weight: root.editing
                                 ? appTheme.fontWeightStrong : appTheme.fontWeightRegular
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideMiddle
                }
            }
        }

        Rectangle {
            anchors.fill: parent
            radius: appTheme.controlRadiusSmall
            color: "transparent"
            border.width: 2
            border.color: root.editing
                         ? appTheme.editorListSelectedInkColor : appTheme.editorListSelectedFillColor
            visible: root.hasFocusFrame
        }
    }
}
