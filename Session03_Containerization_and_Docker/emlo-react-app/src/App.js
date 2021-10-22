import { BrowserRouter, Route, Switch, Redirect } from "react-router-dom";

import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <Switch>
        <Route path="/admin" render={props => <AdminLayout {...props} />} />
        <Redirect from="/" to="/admin/dashboard" />
      </Switch>
    </BrowserRouter>
  //document.getElementById("root")
  );
}

export default App;
