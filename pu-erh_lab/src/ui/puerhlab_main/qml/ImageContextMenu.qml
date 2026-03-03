import QtQuick
import QtQuick.Controls

Menu {
    id: root
    property var actions: []
    signal actionRequested(string actionId)

    function openAt(sceneX, sceneY) {
        x = Math.max(0, sceneX)
        y = Math.max(0, sceneY)
        open()
    }

    Instantiator {
        model: root.actions
        delegate: MenuItem {
            readonly property var actionData: modelData
            text: actionData && actionData.label ? actionData.label : ""
            enabled: !(actionData && actionData.enabled === false)
            onTriggered: root.actionRequested(actionData && actionData.id ? actionData.id : "")
        }
        onObjectAdded: function(index, object) {
            root.insertItem(index, object)
        }
        onObjectRemoved: function(index, object) {
            root.removeItem(object)
            object.destroy()
        }
    }
}
