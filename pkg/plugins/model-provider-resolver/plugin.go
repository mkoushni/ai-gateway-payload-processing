/*
Copyright 2026 The opendatahub.io Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package model_provider_resolver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"path"
	"strings"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	errcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"

	"github.com/opendatahub-io/ai-gateway-payload-processing/pkg/plugins/common/state"
)

const (
	ModelProviderResolverPluginType = "model-provider-resolver"
)

// compile-time type validation
var _ framework.RequestProcessor = &ModelProviderResolverPlugin{}

// ModelProviderResolverFactory defines the factory function for ModelProviderResolverPlugin
func ModelProviderResolverFactory(name string, _ json.RawMessage, handle framework.Handle) (framework.BBRPlugin, error) {
	plugin, err := NewModelProviderResolver(handle.ReconcilerBuilder, handle.ClientReader())
	if err != nil {
		return nil, fmt.Errorf("failed to create plugin '%s' - %w", ModelProviderResolverPluginType, err)
	}

	return plugin.WithName(name), nil
}

func NewModelProviderResolver(reconcilerBuilder func() *builder.Builder, clientReader client.Reader) (*ModelProviderResolverPlugin, error) {
	modelInfoStore := newModelInfoStore()
	reconciler := &externalModelReconciler{
		Reader: clientReader,
		store:  modelInfoStore,
	}

	// Watch ExternalModel CRDs directly (no MaaS dependency)
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(externalModelGVK)

	if err := reconcilerBuilder().For(obj).Complete(reconciler); err != nil {
		return nil, fmt.Errorf("failed to register ExternalModel reconciler for plugin '%s' - %w", ModelProviderResolverPluginType, err)
	}

	return &ModelProviderResolverPlugin{
		typedName:      plugin.TypedName{Type: ModelProviderResolverPluginType, Name: ModelProviderResolverPluginType},
		modelInfoStore: modelInfoStore,
	}, nil
}

// ModelProviderResolverPlugin resolves model names to provider info by watching ExternalModel CRDs.
// It writes the model, provider and credential reference to CycleState for downstream plugins
// (api-translation, api-key-injection).
type ModelProviderResolverPlugin struct {
	typedName      plugin.TypedName
	modelInfoStore *modelInfoStore
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *ModelProviderResolverPlugin) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin instance.
func (p *ModelProviderResolverPlugin) WithName(name string) *ModelProviderResolverPlugin {
	p.typedName.Name = name
	return p
}

// ProcessRequest reads the model name from the request body, resolves the provider
// from the modelInfoStore (populated by ExternalModel reconciler), and writes model, provider
// and credential reference info to CycleState.
func (p *ModelProviderResolverPlugin) ProcessRequest(ctx context.Context, cycleState *framework.CycleState, request *framework.InferenceRequest) error {
	model, ok := request.Body["model"].(string)
	if !ok || model == "" {
		return nil // not an inference request (e.g. API key management, model listing)
	}

	log.FromContext(ctx).V(logutil.VERBOSE).Info("received incoming request", "path", request.Headers[":path"])
	relativePath, err := sanitizePath(request.Headers[":path"])
	if err != nil {
		return errcommon.Error{Code: errcommon.BadRequest, Msg: fmt.Sprintf("invalid request path: %v", err)}
	}

	segments := strings.Split(relativePath, "/")
	if len(segments) < 2 || segments[0] == "" || segments[1] == "" {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("wasn't able to parse namespaced name from path", "path", relativePath)
		return nil
	}

	modelKey := types.NamespacedName{Namespace: segments[0], Name: segments[1]}
	log.FromContext(ctx).V(logutil.VERBOSE).Info("exported namespaced name from path", "key", modelKey)

	externalModelInfo, found := p.modelInfoStore.getModelInfo(modelKey)
	if !found { // info is stored only for external models
		return nil // this is not considered an error, we just need to skip if it's internal model
	}

	if !strings.HasSuffix(relativePath, "chat/completions") { // no support for other input types
		return errcommon.Error{Code: errcommon.BadRequest, Msg: "only /chat/completions input type is supported"}

	}

	// if there's a mismatch it's an error, we don't want to proceed
	if externalModelInfo.targetModel != model {
		return errcommon.Error{Code: errcommon.NotFound, Msg: fmt.Sprintf("model in request body '%s' doesn't match ExternalModel", model)}
	}

	// info of external model written to cycle state for next plugins
	cycleState.Write(state.ProviderKey, externalModelInfo.provider)
	cycleState.Write(state.ModelKey, externalModelInfo.targetModel)
	cycleState.Write(state.CredsRefName, externalModelInfo.secretName)
	cycleState.Write(state.CredsRefNamespace, externalModelInfo.secretNamespace)

	return nil
}

// sanitizePath cleans and validates a relative URL path. It returns an error for paths
// containing null bytes, control characters, path traversal sequences, or invalid encoding.
func sanitizePath(relativeUrlPath string) (string, error) {
	relativeUrlPath = strings.TrimSpace(relativeUrlPath)

	// Null byte protection — must be checked before any further processing.
	if strings.ContainsRune(relativeUrlPath, '\x00') {
		return "", fmt.Errorf("path contains null byte")
	}

	// Control character validation (0x01–0x1F and DEL 0x7F).
	for _, r := range relativeUrlPath {
		if (r >= 0x01 && r <= 0x1F) || r == 0x7F {
			return "", fmt.Errorf("path contains control character 0x%02X", r)
		}
	}

	// Fragment removal — strip '#' and everything after it.
	if idx := strings.IndexByte(relativeUrlPath, '#'); idx >= 0 {
		relativeUrlPath = relativeUrlPath[:idx]
	}

	// Query parameter removal.
	if idx := strings.IndexByte(relativeUrlPath, '?'); idx >= 0 {
		relativeUrlPath = relativeUrlPath[:idx]
	}

	// URL-decode to catch encoding bypasses such as %2e%2e or %2f.
	decoded, err := url.PathUnescape(relativeUrlPath)
	if err != nil {
		return "", fmt.Errorf("path contains invalid percent-encoding: %w", err)
	}

	// Re-check for null bytes that were percent-encoded (e.g. %00).
	if strings.ContainsRune(decoded, '\x00') {
		return "", fmt.Errorf("path contains encoded null byte")
	}

	// Path traversal validation — inspect the decoded path before any normalization
	// so that path.Clean cannot silently absorb traversal sequences.
	// Split on '/' first, then on '\' within each segment to catch Windows-style
	// traversals such as "..\etc\passwd".
	for _, seg := range strings.Split(decoded, "/") {
		for _, bseg := range strings.Split(seg, `\`) {
			if bseg == ".." || bseg == "." {
				return "", fmt.Errorf("path traversal detected")
			}
		}
	}

	// Normalize the validated path.
	normalized := path.Clean("/" + decoded)

	return strings.Trim(normalized, "/"), nil
}
