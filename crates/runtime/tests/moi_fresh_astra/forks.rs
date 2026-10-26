//! New Session fork consumer contract. The ordinary MOI chat baseline stays frozen.
use super::*;
use astra_runtime::server::provider_test_support::{
    ProviderGateway, ProviderResponse, ProviderScript,
};

async fn control(
    app: &Router,
    method: &str,
    path: &str,
    route: &str,
    body: &str,
) -> (StatusCode, Value) {
    response(app, signed_http_request(method, path, route, body.into())).await
}

async fn response(app: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (
        status,
        serde_json::from_slice(&body).expect("control API JSON"),
    )
}

fn signed_in_scope(
    method: &str,
    path: &str,
    route: &str,
    body: &str,
    scope: &str,
) -> Request<Body> {
    let mut request = signed_http_request(method, path, route, body.into());
    let token = request.headers()["authorization"]
        .to_str()
        .unwrap()
        .strip_prefix("Bearer moi-provider-v1.")
        .unwrap();
    let encoded = token.split('.').next().unwrap();
    let mut claims: Value =
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(encoded).unwrap()).unwrap();
    claims["scope"] = json!(scope);
    let encoded = URL_SAFE_NO_PAD.encode(claims.to_string());
    let mut signer = Hmac::<Sha256>::new_from_slice(KEY.as_bytes()).unwrap();
    signer.update(encoded.as_bytes());
    let token = format!(
        "Bearer moi-provider-v1.{encoded}.{}",
        URL_SAFE_NO_PAD.encode(signer.finalize().into_bytes())
    );
    request
        .headers_mut()
        .insert("authorization", token.parse().unwrap());
    request
}

async fn cursor(app: &Router, session: &str, run: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let (status, value) = control(
                app,
                "GET",
                &format!("/sessions/{session}/runs/{run}/cursor"),
                "/sessions/{session_id}/runs/{run_id}/cursor",
                "",
            )
            .await;
            if value["error_code"] == "session_fork_cursor_not_ready" {
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            }
            assert_eq!(status, StatusCode::OK, "cursor: {value}");
            return value;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("committed projection unavailable: session {session}, run {run}"))
}

async fn fork(app: &Router, parent: &str, cursor: &Value, key: &str) -> (Value, String) {
    let path = format!("/sessions/{parent}/forks");
    let payload = json!({"idempotency_key":key,"expected_parent_cursor":cursor,"reason":"MOI selected completed turn"}).to_string();
    let (status, manifest) =
        control(app, "POST", &path, "/sessions/{session_id}/forks", &payload).await;
    assert_eq!(status, StatusCode::OK, "prepare: {manifest}");
    let (status, replay) =
        control(app, "POST", &path, "/sessions/{session_id}/forks", &payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        manifest, replay,
        "same intent cannot allocate a second fork"
    );
    let id = manifest["fork_id"].as_str().unwrap();
    let activate = format!("/sessions/{parent}/forks/{id}/activate-for-continuation");
    let (status, active) = control(
        app,
        "POST",
        &activate,
        "/sessions/{session_id}/forks/{fork_id}/activate-for-continuation",
        "{}",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "activate: {active}");
    assert_eq!(active["manifest"]["state"], "active");
    assert!(
        active.get("writer_lease").is_none(),
        "activation must not grant execution authority"
    );
    let child = manifest["child_key"]["session_id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(active["child_head"]["cursor"]["session_id"], child);
    (active, child)
}

fn answer(text: &str) -> ProviderResponse {
    ProviderResponse::OpenAi(json!({"id":"fork-answer","model":"moi-contract-model",
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "real disposable MatrixOne; provider-signed Session fork contract"]
async fn provider_signed_forks_continue_from_exact_history_without_inherited_authority() {
    assert_eq!(std::env::var("ASTRA_TEST_DB_IT").as_deref(), Ok("1"));
    let mut settings = AppSettings::from_explicit_env().expect("explicit test settings");
    let database = format!("astra_test_moi_fork_{}", Uuid::new_v4().simple());
    eprintln!("MOI fork fixture database: {database}");
    settings.matrixone.database = "mysql".into();
    let mut admin = MySqlConnection::connect(&settings.matrixone.database_url_with_password())
        .await
        .unwrap();
    sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE `{database}`")))
        .execute(&mut admin)
        .await
        .unwrap();
    settings.matrixone.database = database.clone();
    settings.provider_request_auth = vec![ProviderRequestAuthConfig {
        provider: "moi".into(),
        auth_type: "hmac".into(),
        key: KEY.into(),
    }];
    settings.external_providers.clear();
    let gateway_state = Arc::new(Mutex::new(GatewayState::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    settings.memoria.base_url = endpoint.clone();
    let gateway_app = gateway_router(gateway_state.clone());
    let gateway_task = tokio::spawn(async move {
        axum::serve(listener, gateway_app).await.unwrap();
    });
    let state = build_server_state(settings)
        .await
        .expect("fresh Astra bootstrap");
    let app = build_app(state.clone());
    let mut moi = MoiClient::new(false, endpoint, "fork-contract", gateway_state.clone());
    moi.repair_session(&app).await;
    moi.repair_bindings(&app).await;
    moi.history.clear();
    let parent = moi.session.clone().unwrap();
    let provider = ProviderGateway::start(vec![ProviderScript::new(
        "fork turns",
        |r| r.path == "/v1/chat/completions",
        [
            "FIRST-ANSWER",
            "SECOND-ANSWER",
            "THIRD-ANSWER",
            "CHILD-ANSWER",
            "PARENT-ANSWER",
            "SIBLING-ANSWER",
            "CHILD-NEXT",
            "NESTED-ANSWER",
        ]
        .into_iter()
        .map(answer)
        .collect(),
    )])
    .await;
    moi.model_endpoint = Some(format!("{}/v1/chat/completions", provider.base_url));
    let first = moi.chat(&app, "FIRST-FACT", "FIRST-ANSWER").await;
    let first_cursor = cursor(&app, &parent, &first).await;
    assert_eq!(first_cursor["completed_turn"], 1);
    let second = moi.chat(&app, "SECOND-FACT", "SECOND-ANSWER").await;
    moi.chat(&app, "THIRD-FACT", "THIRD-ANSWER").await;
    assert_eq!(
        cursor(&app, &parent, &first).await,
        first_cursor,
        "historical run cannot resolve to latest head"
    );
    let path = format!("/sessions/{parent}/forks");
    let payload = json!({"idempotency_key":"tamper-control","expected_parent_cursor":first_cursor,"reason":"test"}).to_string();
    let mut tampered_body = signed_http_request(
        "POST",
        &path,
        "/sessions/{session_id}/forks",
        payload.clone(),
    );
    *tampered_body.body_mut() = Body::from(payload.replace("tamper-control", "tampered"));
    assert_eq!(
        response(&app, tampered_body).await.0,
        StatusCode::UNAUTHORIZED
    );
    let cursor_path = format!("/sessions/{parent}/runs/{first}/cursor");
    let cursor_route = "/sessions/{session_id}/runs/{run_id}/cursor";
    let mut tampered_path = signed_http_request("GET", &cursor_path, cursor_route, String::new());
    *tampered_path.uri_mut() = format!("/sessions/{parent}/runs/{second}/cursor")
        .parse()
        .unwrap();
    assert_eq!(
        response(&app, tampered_path).await.0,
        StatusCode::UNAUTHORIZED
    );
    let request = signed_http_request("GET", &cursor_path, cursor_route, String::new());
    let headers = request.headers().clone();
    assert_eq!(response(&app, request).await.0, StatusCode::OK);
    let mut replay = Request::builder()
        .method("GET")
        .uri(&cursor_path)
        .body(Body::empty())
        .unwrap();
    *replay.headers_mut() = headers;
    assert_eq!(
        response(&app, replay).await.0,
        StatusCode::UNAUTHORIZED,
        "request ID replay is not an idempotency key"
    );
    let (active, child) = fork(&app, &parent, &first_cursor, "fork-child").await;
    let mut mismatch: Value = serde_json::from_str(&payload).unwrap();
    mismatch["idempotency_key"] = json!("fork-child");
    mismatch["reason"] = json!("different intent");
    let (status, error) = control(
        &app,
        "POST",
        &path,
        "/sessions/{session_id}/forks",
        &mismatch.to_string(),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(error["error_code"], "session_fork_idempotency_mismatch");
    let (status, error) = control(
        &app,
        "GET",
        &format!("/sessions/{child}/runs/{first}/cursor"),
        "/sessions/{session_id}/runs/{run_id}/cursor",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["error_code"], "session_fork_not_found");
    let fork_id = active["manifest"]["fork_id"].as_str().unwrap();
    for (method, path, route, body) in [
        (
            "GET",
            format!("/sessions/{parent}/runs/{first}/cursor"),
            "/sessions/{session_id}/runs/{run_id}/cursor",
            String::new(),
        ),
        (
            "POST",
            format!("/sessions/{parent}/forks"),
            "/sessions/{session_id}/forks",
            payload.clone(),
        ),
        (
            "POST",
            format!("/sessions/{parent}/forks/{fork_id}/activate-for-continuation"),
            "/sessions/{session_id}/forks/{fork_id}/activate-for-continuation",
            "{}".into(),
        ),
    ] {
        let (status, error) = response(
            &app,
            signed_in_scope(method, &path, route, &body, "other-workspace"),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::FORBIDDEN,
            "cross-scope request: {error}"
        );
        assert_eq!(error["error_code"], "session_fork_forbidden");
    }
    let missing_run_path = format!("/sessions/{parent}/runs/{}/cursor", Uuid::new_v4());
    let (status, error) = control(
        &app,
        "GET",
        &missing_run_path,
        "/sessions/{session_id}/runs/{run_id}/cursor",
        "",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["error_code"], "session_fork_not_found");
    for (method, path, route, body) in [
        (
            "GET",
            format!("/sessions/{parent}/forks"),
            "/sessions/{session_id}/forks",
            "",
        ),
        (
            "GET",
            format!("/sessions/{parent}/forks/{fork_id}"),
            "/sessions/{session_id}/forks/{fork_id}",
            "",
        ),
        (
            "POST",
            format!("/sessions/{parent}/forks/{fork_id}/activate"),
            "/sessions/{session_id}/forks/{fork_id}/activate",
            r#"{"placement":"server","writer_ttl_seconds":30}"#,
        ),
        (
            "POST",
            format!("/sessions/{parent}/forks/{fork_id}/abort"),
            "/sessions/{session_id}/forks/{fork_id}/abort",
            r#"{"detail":"legacy admission control","grace_seconds":60}"#,
        ),
    ] {
        let (status, error) = control(&app, method, &path, route, body).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "legacy control must retain its existing authentication: {error}"
        );
    }
    let (_, sibling) = fork(&app, &parent, &first_cursor, "fork-sibling").await;
    assert_ne!(child, sibling);
    assert_eq!(
        provider.requests.lock().await.len(),
        3,
        "fork creates no model calls"
    );
    moi.session = Some(child.clone());
    let child_run = moi.chat(&app, "CHILD-FACT", "CHILD-ANSWER").await;
    let child_cursor = cursor(&app, &child, &child_run).await;
    assert_eq!(
        child_cursor["completed_turn"], 2,
        "child projection starts after its shared prefix"
    );
    moi.session = Some(parent.clone());
    moi.chat(&app, "PARENT-FACT", "PARENT-ANSWER").await;
    moi.session = Some(sibling.clone());
    moi.chat(&app, "SIBLING-FACT", "SIBLING-ANSWER").await;
    moi.session = Some(child.clone());
    let next = moi.chat(&app, "CHILD-NEXT-FACT", "CHILD-NEXT").await;
    assert_eq!(cursor(&app, &child, &next).await["completed_turn"], 3);
    let id = active["manifest"]["fork_id"].as_str().unwrap();
    let (status, replay) = control(
        &app,
        "POST",
        &format!("/sessions/{parent}/forks/{id}/activate-for-continuation"),
        "/sessions/{session_id}/forks/{fork_id}/activate-for-continuation",
        "{}",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        replay["child_head"]["cursor"]["completed_turn"], 3,
        "activation replay must not rewind child"
    );
    let (_, nested) = fork(&app, &child, &child_cursor, "fork-nested").await;
    moi.session = Some(nested.clone());
    let nested_run = moi.chat(&app, "NESTED-FACT", "NESTED-ANSWER").await;
    assert_eq!(
        cursor(&app, &nested, &nested_run).await["completed_turn"],
        3
    );
    assert!(state.drain_background_runs(Duration::from_secs(10)).await);
    // Persisted adversarial states must never resolve to the Session's latest head.
    let pool = state.shared_pool.as_ref().unwrap().get();
    let owner = first_cursor["owner_id"].as_str().unwrap();
    let first_path = format!("/sessions/{parent}/runs/{first}/cursor");
    let cursor_route = "/sessions/{session_id}/runs/{run_id}/cursor";
    sqlx::query(
        "UPDATE agent_runs SET parent_run_id = ?, depth = 1 WHERE user_id = ? AND run_id = ?",
    )
    .bind("delegating-run")
    .bind(owner)
    .bind(&first)
    .execute(pool)
    .await
    .unwrap();
    let (status, value) = control(&app, "GET", &first_path, cursor_route, "").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(value["error_code"], "session_fork_cursor_unsupported");
    sqlx::query("UPDATE agent_runs SET parent_run_id = NULL, depth = 0, status = 'running' WHERE user_id = ? AND run_id = ?")
        .bind(owner).bind(&first).execute(pool).await.unwrap();
    let (status, value) = control(&app, "GET", &first_path, cursor_route, "").await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(value["error_code"], "session_fork_cursor_not_ready");
    let (status, value) = response(
        &app,
        signed_in_scope("GET", &first_path, cursor_route, "", "other-workspace"),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "scope is checked before readiness"
    );
    assert_eq!(value["error_code"], "session_fork_forbidden");
    sqlx::query("UPDATE agent_runs SET status = 'completed' WHERE user_id = ? AND run_id = ?")
        .bind(owner)
        .bind(&first)
        .execute(pool)
        .await
        .unwrap();
    sqlx::query("UPDATE session_transcript_items SET canonical_completed_turn = NULL, canonical_conversation_seq = NULL, canonical_root_hash = NULL WHERE user_id = ? AND run_id = ?")
        .bind(owner).bind(&first).execute(pool).await.unwrap();
    let (status, value) = control(&app, "GET", &first_path, cursor_route, "").await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(value["error_code"], "session_fork_cursor_not_ready");
    sqlx::query("UPDATE session_transcript_items SET canonical_completed_turn = ?, canonical_conversation_seq = ?, canonical_root_hash = ? WHERE user_id = ? AND run_id = ?")
        .bind(first_cursor["completed_turn"].as_i64().unwrap()).bind(first_cursor["conversation_seq"].as_i64().unwrap())
        .bind(first_cursor["canonical_root_hash"].as_str().unwrap()).bind(owner).bind(&first).execute(pool).await.unwrap();
    sqlx::query("UPDATE conversation_manifest_nodes SET reachable = 0 WHERE isolation_domain = 'server' AND owner_user_id = ? AND session_id = ? AND manifest_root = ?")
        .bind(owner).bind(&parent).bind(first_cursor["canonical_root_hash"].as_str().unwrap()).execute(pool).await.unwrap();
    let (status, value) = control(&app, "GET", &first_path, cursor_route, "").await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(value["error_code"], "session_fork_cursor_unavailable");
    sqlx::query("UPDATE conversation_manifest_nodes SET reachable = 1 WHERE isolation_domain = 'server' AND owner_user_id = ? AND session_id = ? AND manifest_root = ?")
        .bind(owner).bind(&parent).bind(first_cursor["canonical_root_hash"].as_str().unwrap()).execute(pool).await.unwrap();
    assert_eq!(cursor(&app, &parent, &first).await, first_cursor);
    // A legacy completed run has no canonical history. Missing scope evidence
    // still denies access before classifying its projection eligibility.
    let (status, legacy) = control(
        &app,
        "POST",
        "/sessions",
        "/sessions",
        &json!({"client_session_ref":format!("legacy-{}", Uuid::new_v4())}).to_string(),
    )
    .await;
    assert!(status == StatusCode::OK || status == StatusCode::CREATED);
    let legacy_session = legacy["session_id"].as_str().unwrap();
    let legacy_run = Uuid::new_v4().to_string();
    sqlx::query("INSERT INTO agent_runs (run_id, user_id, session_id, root_run_id, ancestor_path, status) VALUES (?, ?, ?, ?, ?, 'completed')")
        .bind(&legacy_run).bind(owner).bind(legacy_session).bind(&legacy_run).bind(&legacy_run)
        .execute(pool).await.unwrap();
    let legacy_path = format!("/sessions/{legacy_session}/runs/{legacy_run}/cursor");
    let (status, value) = control(&app, "GET", &legacy_path, cursor_route, "").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(value["error_code"], "session_fork_forbidden");
    sqlx::query("INSERT INTO agent_run_events (id, run_id, event_idx, user_id, session_id, event_type, event_id, event_hash, payload_json) SELECT ?, ?, 0, user_id, ?, event_type, ?, event_hash, payload_json FROM agent_run_events WHERE user_id = ? AND run_id = ? AND event_type = 'run_started'")
        .bind(Uuid::new_v4().to_string()).bind(&legacy_run).bind(legacy_session).bind(Uuid::new_v4().to_string())
        .bind(owner).bind(&first).execute(pool).await.unwrap();
    let (status, value) = control(&app, "GET", &legacy_path, cursor_route, "").await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(value["error_code"], "session_fork_cursor_unsupported");

    let requests = provider.requests.lock().await;
    for (index, included, excluded) in [
        (
            3,
            vec!["FIRST-FACT", "FIRST-ANSWER", "CHILD-FACT"],
            vec!["SECOND-FACT", "THIRD-FACT"],
        ),
        (
            4,
            vec!["FIRST-FACT", "SECOND-FACT", "THIRD-FACT", "PARENT-FACT"],
            vec!["CHILD-FACT"],
        ),
        (
            5,
            vec!["FIRST-FACT", "SIBLING-FACT"],
            vec!["SECOND-FACT", "CHILD-FACT", "PARENT-FACT"],
        ),
        (
            6,
            vec!["FIRST-FACT", "CHILD-FACT", "CHILD-NEXT-FACT"],
            vec!["SIBLING-FACT", "PARENT-FACT"],
        ),
        (
            7,
            vec!["FIRST-FACT", "CHILD-FACT", "NESTED-FACT"],
            vec!["CHILD-NEXT-FACT", "SIBLING-FACT", "PARENT-FACT"],
        ),
    ] {
        let messages = requests[index].body["messages"].to_string();
        for value in included {
            assert!(messages.contains(value), "input {index} missing {value}");
        }
        for value in excluded {
            assert!(!messages.contains(value), "input {index} leaked {value}");
        }
    }
    drop(requests);
    provider.assert_complete();
    assert!(state.drain_background_runs(Duration::from_secs(10)).await);
    state.close_database_pools().await;
    gateway_task.abort();
    sqlx::query(sqlx::AssertSqlSafe(format!("DROP DATABASE `{database}`")))
        .execute(&mut admin)
        .await
        .unwrap();
}
