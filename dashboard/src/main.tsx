import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Classes } from "@blueprintjs/core";
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";
import { App } from "./App";
import "./styles.css";

// On <body> so Blueprint overlays (tooltips, popovers) also render dark.
document.body.classList.add(Classes.DARK);

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
